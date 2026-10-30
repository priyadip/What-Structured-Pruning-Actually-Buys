"""Shared machinery for extending the two MLP compression results to CNNs.

The two studies that live beside this package --

    hidden_layer_compression/   channel budgets from GRIDE on the effective
                                neuron cloud + LS-corrected consumer refit
    patch-space (input) rank reduction, a companion stage not part of this
    repository, chose its rank in
                                the logit-pullback metric (N1_jacobian)

-- are the same two methods that produced the MLP results in `../compression_mlp` and
the input-space stage. Everything mathematical is unchanged; only
the *carrier* changes, from a row of a weight matrix to a convolution filter.

The one idea that makes the port mechanical: a conv layer is a linear map on
**patches**. Unfold its input to `[N*H_out*W_out, C_in*k*k]` and the layer is a
plain matrix multiply, so every closed form from the MLP study applies verbatim
-- with the single difference that the columns now come in *groups* of `k*k`
belonging to the same input channel, so selection has to be group-wise.
"""
