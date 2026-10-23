import numpy as np
import pytest

from predicateneuron.torch import (
    PredicateLayer, MinimalPredicateLayer, predicate_neuron, set_model_hard_forward, set_model_hard_backward
)

torch = pytest.importorskip("torch")


def oracle(x1, x2, theta, T, hard, dy, hard_bw=True):
    """Framework-free re-implementation of the original forward/backward formulas."""
    B, P = x1.shape
    y = np.zeros((B, P));
    dx1 = np.zeros((B, P));
    dx2 = np.zeros((B, P));
    dth = np.zeros(P)
    pos = np.array([0.0, 0.5, 1.0])
    for k in range(P):
        a, b, th = x1[:, k], x2[:, k], theta[k]
        vals = np.stack([a * b, a + b - a * b, a + b - 2 * a * b], -1)
        g = 0 if th < 0.25 else (1 if th < 0.75 else 2)
        logits = -(th - pos) ** 2 / T
        p = np.exp(logits - logits.max());
        p /= p.sum()
        y[:, k] = vals[:, g] if hard else vals @ p
        dz = -2 * (th - pos) / T
        dp = p * (dz - (p * dz).sum())
        dth[k] = (dy[:, k] * (vals @ dp)).sum()
        if hard_bw:
            d1 = [b, 1 - b, 1 - 2 * b][g];
            d2 = [a, 1 - a, 1 - 2 * a][g]
        else:
            d1 = np.stack([b, 1 - b, 1 - 2 * b], -1) @ p
            d2 = np.stack([a, 1 - a, 1 - 2 * a], -1) @ p
        dx1[:, k] = dy[:, k] * d1;
        dx2[:, k] = dy[:, k] * d2
    return y, dx1, dx2, dth


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("hard_bw", [True, False])
def test_functional_matches_original_formulas(hard, hard_bw):
    rng = np.random.default_rng(0)
    B, P, T = 16, 5, 0.3
    x1, x2 = rng.random((B, P)), rng.random((B, P))
    theta = np.array([0.05, 0.3, 0.5, 0.7, 0.95]);
    w = rng.normal(size=(B, P))
    t1 = torch.tensor(x1, requires_grad=True);
    t2 = torch.tensor(x2, requires_grad=True)
    tt = torch.tensor(theta, requires_grad=True)
    y = predicate_neuron(t1, t2, tt, T, hard, hard_bw)
    (y * torch.tensor(w)).sum().backward()
    ry, rd1, rd2, rdt = oracle(x1, x2, theta, T, hard, w, hard_bw)
    np.testing.assert_allclose(y.detach().numpy(), ry, atol=1e-10)
    np.testing.assert_allclose(t1.grad.numpy(), rd1, atol=1e-10)
    np.testing.assert_allclose(t2.grad.numpy(), rd2, atol=1e-10)
    np.testing.assert_allclose(tt.grad.numpy(), rdt, atol=1e-8)


@pytest.mark.parametrize("Layer,out", [(PredicateLayer, 10), (MinimalPredicateLayer, 3)])
def test_layer_shapes_and_switch(Layer, out):
    model = torch.nn.Sequential(Layer(5, hard_forward=True), torch.nn.Linear(out, 1))
    set_model_hard_forward(model, False)
    assert not bool(model[0].hard_forward)
    model(torch.rand(4, 5)).sum().backward()
    assert model[0].theta.grad is not None
    assert model[0](torch.rand(4, 5)).shape == (4, out)


@pytest.mark.parametrize("hard", [True, False])
def test_torch_matches_keras(hard):
    keras = pytest.importorskip("keras")
    from predicateneuron.keras import PredicateLayer as KP
    x = np.random.rand(6, 5).astype("float32")
    theta = np.linspace(0.05, 0.95, 10).astype("float32")
    k = KP(hard_forward=hard);
    k.build((None, 5));
    k.theta.assign(theta)
    t = PredicateLayer(5, hard_forward=hard)
    with torch.no_grad():
        t.theta.copy_(torch.tensor(theta))
    np.testing.assert_allclose(np.array(k(x)), t(torch.tensor(x)).detach().numpy(), rtol=1e-5, atol=1e-6)


def test_soft_backward_is_true_autograd_of_mixture():
    """hard_backward=False + hard_forward=False must equal plain backprop through the mixture."""
    torch.manual_seed(0)
    x1 = torch.rand(8, 3, dtype=torch.float64, requires_grad=True)
    x2 = torch.rand(8, 3, dtype=torch.float64, requires_grad=True)
    th = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, requires_grad=True)
    y = predicate_neuron(x1, x2, th, 0.3, False, False)
    g1, g2 = torch.autograd.grad(y.sum(), [x1, x2])

    pos = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    vals = torch.stack([x1 * x2, x1 + x2 - x1 * x2, x1 + x2 - 2 * x1 * x2], -1)
    p = torch.softmax(-(th.unsqueeze(-1) - pos) ** 2 / 0.3, -1)
    ref = (vals * p).sum(-1)
    r1, r2 = torch.autograd.grad(ref.sum(), [x1, x2])
    torch.testing.assert_close(y, ref)
    torch.testing.assert_close(g1, r1);
    torch.testing.assert_close(g2, r2)


def test_set_model_hard_backward():
    model = torch.nn.Sequential(PredicateLayer(4, hard_forward=True), MinimalPredicateLayer(6, hard_forward=True))
    set_model_hard_backward(model, False)
    assert not bool(model[0].hard_backward) and not bool(model[1].hard_backward)
    assert bool(model[0].hard_forward)  # untouched
