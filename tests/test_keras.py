import numpy as np
import pytest

keras = pytest.importorskip("keras")
from predicateneuron.keras import PredicateLayer, MinimalPredicateLayer, set_model_hard_forward, set_model_hard_backward


@pytest.mark.parametrize("Layer,out", [(PredicateLayer, 10), (MinimalPredicateLayer, 3)])
def test_shape_and_save_load(tmp_path, Layer, out):
    model = keras.Sequential([keras.Input((5,)), Layer(temperature=0.2)])
    x = np.random.rand(4, 5).astype("float32")
    assert model(x).shape == (4, out)
    model.save(tmp_path / "m.keras")
    loaded = keras.saving.load_model(tmp_path / "m.keras")
    np.testing.assert_allclose(model(x), loaded(x), rtol=1e-6)


def test_hard_soft_switch_and_training():
    model = keras.Sequential([keras.Input((4,)), PredicateLayer(), keras.layers.Dense(1)])
    x = np.random.rand(32, 4).astype("float32"); y = np.random.rand(32, 1).astype("float32")
    model.compile("adam", "mse")
    set_model_hard_forward(model, False)
    model.fit(x, y, epochs=1, verbose=0)
    layer = model.layers[0]
    assert layer._hard_forward is False and float(layer.hard_forward.numpy()) == 0.0
    assert not np.allclose(layer.theta.numpy(), 0.5)    # theta received gradient


def test_hard_backward_switch_and_config(tmp_path):
    model = keras.Sequential([keras.Input((4,)), PredicateLayer(hard_backward=False)])
    layer = model.layers[0]
    assert layer._hard_backward is False and float(layer.hard_backward.numpy()) == 0.0
    set_model_hard_backward(model, True)
    assert layer._hard_backward is True and float(layer.hard_backward.numpy()) == 1.0
    model.save(tmp_path / "m.keras")
    assert keras.saving.load_model(tmp_path / "m.keras").layers[0]._hard_backward is True


def test_soft_backward_exact_gradient_tf_backend():
    if keras.backend.backend() != "tensorflow":
        pytest.skip("uses tf.GradientTape")
    import tensorflow as tf
    from predicateneuron.keras import predicate_neuron
    x1 = tf.constant(np.random.rand(8, 3)); x2 = tf.constant(np.random.rand(8, 3))
    th = tf.constant([0.1, 0.5, 0.9], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch([x1, x2])
        y = predicate_neuron(x1, x2, th, 0.3, False, False)
    g1 = tape.gradient(y, x1)
    pos = tf.constant([0.0, 0.5, 1.0], tf.float64)
    p = tf.nn.softmax(-tf.square(th[:, None] - pos) / 0.3, -1)
    d1 = tf.stack([x2, 1 - x2, 1 - 2 * x2], -1)
    np.testing.assert_allclose(g1, tf.reduce_sum(d1 * p, -1), atol=1e-10)
