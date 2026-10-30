# One-off training runner

`resnet50_imagenet32.py` trained `imagenet32_resnet50`, one of the sixteen
convolutional checkpoints in the paper. It exists separately from
`trainer/train.py` because that run was restarted from a partial state after an
interruption, rather than launched cleanly.

It is included for provenance: every checkpoint the paper uses should have the
code that produced it somewhere in this supplement. It is not part of the
reproduction path, and `trainer/train.py` will train the same model from scratch.


