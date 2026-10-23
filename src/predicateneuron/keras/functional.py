from keras import ops


def predicate_neuron(x1, x2, theta, temperature=0.06, hard_forward=True, hard_backward=True):
    """Differentiable predicate neuron, element-wise over broadcastable x1, x2, theta.

    Forward : theta selects intersection (<0.25), union (<0.75) or symmetric
              difference (otherwise) when `hard_forward`, else a softmax mixture
              of the three over positions [0, 0.5, 1].
    Backward:
      d/dtheta  always from the soft (softmax) surrogate.
      d/dx1,x2  hard_backward=True : derivative of the hard-selected operation
                                     (behaviour of the original paper code).
                hard_backward=False: exact derivative of the soft mixture,
                                     sum_k p_k * d(op_k)/dx.
    """

    sg = ops.stop_gradient

    intersection = x1 * x2
    union = x1 + x2 - intersection
    symmetric_diff = x1 + x2 - 2.0 * intersection

    t = sg(theta)
    gate = ops.where(t < 0.25, 0, ops.where(t < 0.75, 1, 2))
    y_hard = ops.where(gate == 0, intersection, ops.where(gate == 1, union, symmetric_diff))

    positions = ops.cast(ops.convert_to_tensor([0.0, 0.5, 1.0]), x1.dtype)
    values = ops.stack([intersection, union, symmetric_diff], axis=-1)
    logits = -ops.square(ops.expand_dims(theta, -1) - positions) / temperature
    p = ops.softmax(logits, axis=-1)
    y_soft_theta = ops.sum(sg(values) * p, axis=-1)
    y_soft_x = ops.sum(values * sg(p), axis=-1)

    forward = ops.where(hard_forward, y_hard, y_soft_theta)
    y_x = ops.where(hard_backward, y_hard, y_soft_x)
    return sg(forward) + (y_soft_theta - sg(y_soft_theta)) + (y_x - sg(y_x))
