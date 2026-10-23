# predicateneuron

A differentiable **predicate neuron** and ready-made **predicate layers** for Keras, TensorFlow and PyTorch.

The neuron combines two inputs in [0, 1] with a fuzzy-logic operation (intersection, union or symmetric difference). A
single learnable parameter, `theta`, decides which operation is used, so the network learns which logical relation to
apply to each pair of features.

> Code accompanying the paper: **TODO: title** ([link](https://TODO)).

## Installation

```bash
pip install "predicateneuron[keras]"        # Keras 3 (TensorFlow, PyTorch or JAX backend)
pip install "predicateneuron[tensorflow]"   # TensorFlow >= 2.16
pip install "predicateneuron[torch]"        # PyTorch
pip install "predicateneuron[all]"          # everything
```

Only the framework you need has to be installed. Importing `predicateneuron` itself pulls in none of them.

## Quick start

**Keras / TensorFlow**

```python
import keras
from predicateneuron.keras import PredicateLayer  # or: from predicateneuron.tf import ...

model = keras.Sequential([
    keras.Input((8,)),
    PredicateLayer(
        temperature=0.06, theta_initializer=keras.initializers.Constant(0.5),
        hard_forward=False, hard_backward=False
    ),
    keras.layers.Dense(1, activation="sigmoid"),
])
```

**PyTorch**

```python
import torch
from predicateneuron.torch import PredicateLayer

model = torch.nn.Sequential(
    PredicateLayer(in_features=8, temperature=0.06, theta_init=0.5,
                   hard_forward=False, hard_backward=False),
    torch.nn.Linear(28, 1),  # 8 inputs -> 8*7/2 = 28 pairs
    torch.nn.Sigmoid(),
)
```

Inputs are expected to lie in [0, 1], for example sigmoid outputs or normalized features.

## The neuron

For inputs `x1`, `x2` and a learnable `theta`:

| `theta`        | operation            | formula              |
|----------------|----------------------|----------------------|
| `< 0.25`       | intersection         | `x1·x2`              |
| `0.25 … 0.75`  | union                | `x1 + x2 − x1·x2`    |
| `≥ 0.75`       | symmetric difference | `x1 + x2 − 2·x1·x2`  |

The function is also available directly: `from predicateneuron.keras import predicate_neuron` (or `.torch`).

### Forward and backward modes

Two independent switches control how the neuron behaves.

| Flag                      | `True`                                               | `False` (default)                                                   |
|---------------------------|------------------------------------------------------|---------------------------------------------------------------------|
| `hard_forward`            | output uses the single operation selected by `theta` | output is a softmax mixture of the three operations (`temperature`) |
| `hard_backward`           | input gradients from the selected operation only     | input gradients from the exact derivative of the soft mixture       |

The gradient with respect to `theta` always comes from the softmax surrogate, in every mode. All four combinations are
valid. The defaults are (`hard_forward=False`, `hard_backward=False`).

## Layers

| Layer                       | Pairs                                              | Output size  |
|-----------------------------|----------------------------------------------------|--------------|
| `PredicateLayer`            | every unique pair `(i, j)`, `i < j`                | `n(n−1)/2`   |
| `MinimalPredicateLayer`     | disjoint pairs `(0,1), (2,3), …`; odd last input is passed through | `⌈n/2⌉` |
| `MinimalJointPredicateLayer`| overlapping circular pairs `(0,1), (1,2), …, (n−1,0)` | `n`       |

Every layer owns one `theta` per pair and accepts inputs of any rank, with the features on the last axis.

## Switching modes

Each layer keeps its own flags, so layers in one model can run in different modes.

```python
from predicateneuron.keras import set_model_hard_forward, set_model_hard_backward

# a single layer
layer.set_hard_forward(False)
layer.set_hard_backward(False)

# every predicate layer in a model (including nested ones)
set_model_hard_forward(model, False)  # e.g. soft warm-up
set_model_hard_backward(model, False)
set_model_hard_forward(model, True)  # switch back to hard gates

# mixed: all soft except one layer
set_model_hard_forward(model, False)
model.get_layer("last_predicate").set_hard_forward(True)
```

The flags are stored as variables (Keras) or buffers (PyTorch), so they can be changed during training without
rebuilding the model. In Keras they are also part of `get_config`, so saved models restore their mode.

## Notes

- Keras support targets **Keras 3**. TensorFlow 2.15 and earlier ship Keras 2 as `tf.keras`, which is not supported.
- Both `theta` and the three candidate operations are differentiated separately by design (see *Forward and backward
  modes*), so soft mode is not plain backpropagation through the mixture unless `hard_backward=False`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

Apache 2.0