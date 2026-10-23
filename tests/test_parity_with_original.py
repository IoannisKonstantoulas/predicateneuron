"""The port must reproduce the ORIGINAL tf.custom_gradient neuron exactly:
same forward values and same gradients (x1, x2, theta), in hard and soft mode."""
import importlib.util
import pathlib

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")
if keras.backend.backend() != "tensorflow":
    pytest.skip("parity test needs KERAS_BACKEND=tensorflow", allow_module_level=True)

from predicateneuron.keras import predicate_neuron as new_neuron

_ref_path = pathlib.Path(__file__).parents[1] / "reference" / "original_tf.py"
_spec = importlib.util.spec_from_file_location("original_tf", _ref_path)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("theta", [0.05, 0.3, 0.5, 0.7, 0.95])
@pytest.mark.parametrize("temperature", [0.1, 1.0])
def test_forward_and_gradients_match(hard, theta, temperature):
    rng = np.random.default_rng(0)
    x1 = rng.random(16).astype("float32")
    x2 = rng.random(16).astype("float32")
    w = rng.normal(size=16).astype("float32")

    a1, a2, at = tf.constant(x1), tf.constant(x2), tf.constant(theta, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([a1, a2, at])
        y_ref = ref.predicate_neuron(a1, a2, at, temperature, tf.constant(hard))
        loss = tf.reduce_sum(y_ref * w)
    g_ref = tape.gradient(loss, [a1, a2, at])

    b1, b2 = tf.constant(x1[:, None]), tf.constant(x2[:, None])
    bt = tf.constant([theta], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([b1, b2, bt])
        y_new = new_neuron(b1, b2, bt, temperature, hard)
        loss = tf.reduce_sum(y_new[:, 0] * w)
    g_new = tape.gradient(loss, [b1, b2, bt])

    np.testing.assert_allclose(y_new[:, 0], y_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g_new[0][:, 0], g_ref[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(g_new[1][:, 0], g_ref[1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(g_new[2][0], g_ref[2], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("hard", [True, False])
def test_layers_match_original(hard):
    rng = np.random.default_rng(1)
    x = rng.random((8, 5)).astype("float32")
    from predicateneuron.keras import PredicateLayer, MinimalPredicateLayer
    for New, Old in [(PredicateLayer, ref.PredicateLayer),
                     (MinimalPredicateLayer, ref.MinimalPredicateLayer)]:
        old, new = Old(temperature=0.1, hard_forward=hard), New(temperature=0.1, hard_forward=hard)
        theta = np.linspace(0.05, 0.95, 10 if Old is ref.PredicateLayer else 2).astype("float32")
        old.build((None, 5));
        new.build((None, 5))
        old.theta.assign(theta);
        new.theta.assign(theta)
        np.testing.assert_allclose(new(x), old(x), rtol=1e-6, atol=1e-6)
