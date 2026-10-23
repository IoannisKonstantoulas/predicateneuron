import keras
import numpy as np
from keras import ops

from .functional import predicate_neuron


def set_model_hard_forward(model, value):
    """Switch predicate layer between hard/soft forward pass."""
    for layer in getattr(model, "layers", []):
        if isinstance(layer, _BasePredicateLayer):
            layer.set_hard_forward(value)
        set_model_hard_forward(layer, value)


def set_model_hard_backward(model, value):
    """Switch predicate layer between operation-selection (True) and weighted-aggregate (False) of input gradients."""
    for layer in getattr(model, "layers", []):
        if isinstance(layer, _BasePredicateLayer):
            layer.set_hard_backward(value)
        set_model_hard_backward(layer, value)


class _BasePredicateLayer(keras.layers.Layer):
    def __init__(
            self, temperature=0.06, theta_initializer=keras.initializers.Constant(0.5),
            hard_forward=False, hard_backward=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self._hard_forward = bool(hard_forward)
        self._hard_backward = bool(hard_backward)
        self.theta_initializer = keras.initializers.get(theta_initializer)

    def _make_weights(self, n_pairs):
        self.theta = self.add_weight(
            name="theta", shape=(n_pairs,),
            initializer=self.theta_initializer, trainable=True)
        self.hard_forward = self.add_weight(
            name="hard_forward", shape=(),
            initializer=keras.initializers.Constant(float(self._hard_forward)),
            trainable=False)
        self.hard_backward = self.add_weight(
            name="hard_backward", shape=(),
            initializer=keras.initializers.Constant(float(self._hard_backward)),
            trainable=False)

    def set_hard_forward(self, value):
        self._hard_forward = bool(value)
        if getattr(self, "hard_forward", None) is not None:
            self.hard_forward.assign(float(self._hard_forward))

    def set_hard_backward(self, value):
        self._hard_backward = bool(value)
        if getattr(self, "hard_backward", None) is not None:
            self.hard_backward.assign(float(self._hard_backward))

    def _hard_fw(self):
        return ops.greater(self.hard_forward, 0.5)

    def _hard_bw(self):
        return ops.greater(self.hard_backward, 0.5)

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "hard_forward": self._hard_forward,
            "hard_backward": self._hard_backward,
            "theta_initializer": keras.initializers.serialize(self.theta_initializer),
        })
        return config


@keras.saving.register_keras_serializable(package="predicateneuron")
class PredicateLayer(_BasePredicateLayer):
    """Pairwise fully connected: One predicate neuron for every unique pair (i<j) of inputs:
    (..., n) -> (...,n(n-1)/2)."""

    def build(self, input_shape):
        n = input_shape[-1]
        if n is None:
            raise ValueError("The last input dimension must be known.")
        n = int(n)
        if n < 2:
            raise ValueError("PredicateLayer needs at least 2 input features.")
        self.n_inputs = n
        i, j = np.triu_indices(n, k=1)
        self._idx_i, self._idx_j = i, j
        self.n_pairs = len(i)
        self._make_weights(self.n_pairs)

    def call(self, x):
        x1 = ops.take(x, self._idx_i, axis=-1)
        x2 = ops.take(x, self._idx_j, axis=-1)
        return predicate_neuron(x1, x2, self.theta, self.temperature,
                                self._hard_fw(), self._hard_bw())

    def compute_output_shape(self, input_shape):
        n = int(input_shape[-1])
        return tuple(input_shape[:-1]) + (n * (n - 1) // 2,)


@keras.saving.register_keras_serializable(package="predicateneuron")
class MinimalPredicateLayer(_BasePredicateLayer):
    """Neurons on disjoint consecutive pairs (0,1),(2,3),...,(n-1,n). Possible leftover odd input passes through."""

    def build(self, input_shape):
        n = input_shape[-1]
        if n is None:
            raise ValueError("The last input dimension must be known.")
        self.n_inputs = int(n)
        self.n_pairs = self.n_inputs // 2
        self._make_weights(self.n_pairs)

    def call(self, x):
        if self.n_pairs == 0:
            return x
        m = 2 * self.n_pairs
        y = predicate_neuron(
            x[..., 0:m:2], x[..., 1:m:2], self.theta, self.temperature, self._hard_fw(), self._hard_bw()
        )
        if self.n_inputs % 2 == 1:
            y = ops.concatenate([y, x[..., -1:]], axis=-1)
        return y

    def compute_output_shape(self, input_shape):
        n = int(input_shape[-1])
        return tuple(input_shape[:-1]) + (n // 2 + n % 2,)


@keras.saving.register_keras_serializable(package="predicateneuron")
class MinimalJointPredicateLayer(_BasePredicateLayer):
    """Neurons on joint circular pairs (0,1),(1,2),...,(n-2,n-1),(n-1,0)."""

    def build(self, input_shape):
        n = input_shape[-1]
        if n is None:
            raise ValueError("The last input dimension must be known.")
        if n < 2:
            raise ValueError("MinimalJointPredicateLayer needs at least 2 input features.")
        self.n_inputs = int(n)
        self.n_pairs = int(n)
        self._make_weights(self.n_pairs)

    def call(self, x):
        x_next = ops.concatenate([x[..., 1:], x[..., :1]], axis=-1)
        return predicate_neuron(x, x_next, self.theta, self.temperature, self._hard_fw(), self._hard_bw())

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (int(input_shape[-1]),)
