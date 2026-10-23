import torch


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

    intersection = x1 * x2
    union = x1 + x2 - intersection
    symmetric_diff = x1 + x2 - 2.0 * intersection

    t = theta.detach()
    gate = (t >= 0.25).long() + (t >= 0.75).long()
    y_hard = torch.where(gate == 0, intersection, torch.where(gate == 1, union, symmetric_diff))

    positions = torch.tensor([0.0, 0.5, 1.0], dtype=x1.dtype, device=x1.device)
    values = torch.stack([intersection, union, symmetric_diff], dim=-1)
    logits = -torch.square(theta.unsqueeze(-1) - positions) / temperature
    p = torch.softmax(logits, dim=-1)
    y_soft_theta = (values.detach() * p).sum(dim=-1)
    y_soft_x = (values * p.detach()).sum(dim=-1)

    hf = torch.as_tensor(hard_forward, dtype=torch.bool, device=x1.device)
    hb = torch.as_tensor(hard_backward, dtype=torch.bool, device=x1.device)
    forward = torch.where(hf, y_hard, y_soft_theta)
    y_x = torch.where(hb, y_hard, y_soft_x)
    return (
            forward.detach()
            + (y_soft_theta - y_soft_theta.detach())
            + (y_x - y_x.detach())
    )
