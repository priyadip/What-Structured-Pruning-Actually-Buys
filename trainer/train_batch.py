"""Trains multiple (dataset, model) pairs across all visible GPUs using a
worker pool with a bounded number of concurrent jobs per GPU.

Each GPU hosts `slots_per_gpu` worker processes that pull jobs from a shared
queue. Bounding concurrency (rather than launching every job at once) keeps
each running job at high GPU utilization instead of many jobs thrashing the
same SMs — better wall-clock throughput for compute-heavy conv nets.

Usage:
    python -m trainer.train_batch
    python -m trainer.train_batch --jobs-json my_jobs.json --num-gpus 2 --slots-per-gpu 2
"""

import argparse
import json
import multiprocessing as mp
import os
import traceback

# 12 conv models: 2 VGG (vgg11_bn = paper-canonical, vgg16_bn = higher capacity)
# and 2 ResNet (resnet18 = paper-canonical, resnet34 = higher capacity) on each
# of FashionMNIST / CIFAR10 / CIFAR100. Architectures are the ones ported from
# Intra-Fusion, so these checkpoints are directly usable for that paper's OT
# pruning later. FashionMNIST (1x28x28) is auto-adapted to 3x32x32 by the
# trainer so the 3-channel VGG/ResNet stem accepts it. Conv nets default to
# SGD+momentum+weight_decay (set in trainer/train.py); patience is set >= epochs
# so the cosine LR schedule always completes (its final decay gives an accuracy
# bump we don't want early-stopping to cut off), and the best-val-acc checkpoint
# is saved regardless.
_ARCHS = ["vgg11_bn", "vgg16_bn", "resnet18", "resnet34"]
_DATASETS = ["FashionMNIST", "CIFAR10", "CIFAR100"]
_COMMON = {"epochs": 150, "batch_size": 128, "patience": 200,
           "train_kwargs": {"label_smoothing": 0.1}}

DEFAULT_JOBS = [
    {"dataset": ds, "model": arch, **_COMMON}
    for ds in _DATASETS
    for arch in _ARCHS
]


def _worker(gpu_id: int, job_queue: mp.Queue, result_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Deferred so this process only initializes CUDA after being pinned to its
    # assigned GPU via the env var above.
    from trainer.train import train

    while True:
        job = job_queue.get()
        if job is None:  # sentinel: no more work for this worker
            break
        try:
            ckpt_path, best_val_acc = train(
                job["dataset"], job["model"], epochs=job.get("epochs"), batch_size=job.get("batch_size"),
                lr=job.get("lr"), patience=job.get("patience"), device="cuda",
                augment=job.get("augment", True),
                **job.get("train_kwargs", {}), **job.get("model_kwargs", {}),
            )
            result_queue.put({**job, "gpu": gpu_id, "ckpt_path": str(ckpt_path),
                              "best_val_acc": best_val_acc})
        except Exception:
            result_queue.put({**job, "gpu": gpu_id, "best_val_acc": None,
                              "error": traceback.format_exc()})


def run_batch(jobs: list, num_gpus: int = None, slots_per_gpu: int = 2) -> list:
    import torch
    num_gpus = num_gpus or max(torch.cuda.device_count(), 1)

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for job in jobs:
        job_queue.put(job)

    workers = []
    for gpu in range(num_gpus):
        for _ in range(slots_per_gpu):
            job_queue.put(None)  # one stop-sentinel per worker (drained after all real jobs)
            p = ctx.Process(target=_worker, args=(gpu, job_queue, result_queue))
            p.start()
            workers.append(p)

    results = [result_queue.get() for _ in jobs]  # blocks until every job has reported in
    for p in workers:
        p.join()
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train multiple (dataset, model) pairs in parallel.")
    p.add_argument("--jobs-json", default=None,
                   help="Path to a JSON file with a list of job dicts (dataset, model, epochs, "
                        "batch_size, patience, lr, train_kwargs, model_kwargs). Defaults to this "
                        "project's 12-model conv-net batch.")
    p.add_argument("--num-gpus", type=int, default=None)
    p.add_argument("--slots-per-gpu", type=int, default=2,
                   help="Concurrent training jobs per GPU (default 2).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    jobs = DEFAULT_JOBS
    if args.jobs_json:
        with open(args.jobs_json) as f:
            jobs = json.load(f)

    results = run_batch(jobs, num_gpus=args.num_gpus, slots_per_gpu=args.slots_per_gpu)
    results.sort(key=lambda r: (r["dataset"], r["model"]))

    print("\n=== Batch training summary ===")
    for r in results:
        if r.get("best_val_acc") is None:
            print(f"{r['dataset']:<14} {r['model']:<12} FAILED\n{r.get('error', '')}")
        else:
            print(f"{r['dataset']:<14} {r['model']:<12} best_val_acc={r['best_val_acc']:.4f}  {r['ckpt_path']}")
