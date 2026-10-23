"""TensorFlow entry point: the Keras 3 layers are native tf.keras layers when the
Keras backend is TensorFlow (the default; TF >= 2.16). Make sure KERAS_BACKEND is
unset or 'tensorflow' if you pass raw tf tensors to `predicate_neuron`."""
from .keras import (
    predicate_neuron, PredicateLayer, MinimalPredicateLayer, MinimalJointPredicateLayer,
    set_model_hard_forward, set_model_hard_backward)

__all__ = [
    "predicate_neuron",
    "PredicateLayer",
    "MinimalPredicateLayer",
    "MinimalJointPredicateLayer",
    "set_model_hard_forward",
    "set_model_hard_backward"
]
