"""predicateneuron: framework-specific code lives in subpackages, so importing
`predicateneuron` never forces Keras, TensorFlow or PyTorch to be installed.

from predicateneuron.keras import PredicateLayer, MinimalPredicateLayer, MinimalJointPredicateLayer   # Keras 3
from predicateneuron.tf    import PredicateLayer, MinimalPredicateLayer, MinimalJointPredicateLayer   # tf.keras users
from predicateneuron.torch import PredicateLayer, MinimalPredicateLayer, MinimalJointPredicateLayer   # PyTorch
"""
__version__ = "0.1.4"
