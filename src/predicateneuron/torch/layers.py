import torch
from torch import nn

from .functional import predicate_neuron


def set_model_hard_forward(model, value):
    """Switch predicate layer between hard/soft forward pass."""
    for m in model.modules():
        if isinstance(m, (PredicateLayer, MinimalPredicateLayer, MinimalJointPredicateLayer)):
            m.set_hard_forward(value)


def set_model_hard_backward(model, value):
    """Switch predicate layer between operation-selection (True) and weighted-aggregate (False) of input gradients."""
    for m in model.modules():
        if isinstance(m, (PredicateLayer, MinimalPredicateLayer, MinimalJointPredicateLayer)):
            m.set_hard_backward(value)


class PredicateLayer(nn.Module):
    """Pairwise fully connected: One predicate neuron for every unique pair (i<j) of inputs:
    (..., n) -> (...,n(n-1)/2)."""

    def __init__(self, in_features, temperature=0.06, theta_init=0.5, hard_forward=False, hard_backward=False):
        super().__init__()
        if in_features < 2:
            raise ValueError("PredicateLayer needs at least 2 input features.")
        self.in_features = in_features
        self.temperature = temperature
        self.theta_init = theta_init
        i, j = torch.triu_indices(in_features, in_features, offset=1)
        self.register_buffer("idx_i", i, persistent=False)
        self.register_buffer("idx_j", j, persistent=False)
        self.n_pairs = i.numel()
        self.theta = nn.Parameter(torch.empty(self.n_pairs))
        self.register_buffer("hard_forward", torch.tensor(bool(hard_forward)))
        self.register_buffer("hard_backward", torch.tensor(bool(hard_backward)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.theta, self.theta_init)

    def set_hard_forward(self, value):
        self.hard_forward.fill_(bool(value))

    def set_hard_backward(self, value):
        self.hard_backward.fill_(bool(value))

    def forward(self, x):
        return predicate_neuron(
            x[..., self.idx_i], x[..., self.idx_j], self.theta, self.temperature, self.hard_forward, self.hard_backward
        )

    def extra_repr(self):
        return (f"in_features={self.in_features}, n_pairs={self.n_pairs}, "
                f"temperature={self.temperature}, hard_forward={bool(self.hard_forward)}, "
                f"hard_backward={bool(self.hard_backward)}")


class MinimalPredicateLayer(nn.Module):
    """Neurons on disjoint consecutive pairs (0,1),(2,3),...,(n-1,n). Possible leftover odd input passes through."""

    def __init__(self, in_features, temperature=0.06, theta_init=0.5, hard_forward=False, hard_backward=False):
        super().__init__()
        self.in_features = in_features
        self.temperature = temperature
        self.theta_init = theta_init
        self.n_pairs = in_features // 2
        self.theta = nn.Parameter(torch.empty(self.n_pairs))
        self.register_buffer("hard_forward", torch.tensor(bool(hard_forward)))
        self.register_buffer("hard_backward", torch.tensor(bool(hard_backward)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.theta, self.theta_init)

    def set_hard_forward(self, value):
        self.hard_forward.fill_(bool(value))

    def set_hard_backward(self, value):
        self.hard_backward.fill_(bool(value))

    def forward(self, x):
        if self.n_pairs == 0:
            return x
        m = 2 * self.n_pairs
        y = predicate_neuron(
            x[..., 0:m:2], x[..., 1:m:2], self.theta, self.temperature, self.hard_forward, self.hard_backward
        )
        if self.in_features % 2 == 1:
            y = torch.cat([y, x[..., -1:]], dim=-1)
        return y

    def extra_repr(self):
        return (f"in_features={self.in_features}, n_pairs={self.n_pairs}, "
                f"temperature={self.temperature}, hard_forward={bool(self.hard_forward)}, "
                f"hard_backward={bool(self.hard_backward)}")

class MinimalJointPredicateLayer(nn.Module):
    """Neurons on joint circular pairs (0,1),(1,2),...,(n-2,n-1),(n-1,0)."""

    def __init__(self, in_features, temperature=0.1, theta_init=0.5, hard_forward=False, hard_backward=False):
        super().__init__()
        if in_features < 2:
            raise ValueError("MinimalJointPredicateLayer needs at least 2 input features.")
        self.in_features = in_features
        self.temperature = temperature
        self.theta_init = theta_init
        self.n_pairs = in_features
        self.theta = nn.Parameter(torch.empty(self.n_pairs))
        self.register_buffer("hard_forward", torch.tensor(bool(hard_forward)))
        self.register_buffer("hard_backward", torch.tensor(bool(hard_backward)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.theta, self.theta_init)

    def set_hard_forward(self, value):
        self.hard_forward.fill_(bool(value))

    def set_hard_backward(self, value):
        self.hard_backward.fill_(bool(value))

    def forward(self, x):
        x_next = torch.cat([x[..., 1:], x[..., :1]], dim=-1)
        return predicate_neuron(x, x_next, self.theta, self.temperature, self.hard_forward, self.hard_backward)
