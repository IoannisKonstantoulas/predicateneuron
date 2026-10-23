"""ORIGINAL implementation used in the paper experiments (tf.keras / Keras 2).
Kept verbatim as the reference oracle for tests/test_parity_with_original.py.
Do not edit."""
import tensorflow as tf


def set_model_hard_forward(model, value):
    for layer in model.layers:
        if isinstance(layer, (PredicateLayer, MinimalPredicateLayer)):
            layer.set_hard_forward(value)

@tf.custom_gradient
def predicate_neuron(x1, x2, theta, temperature=1.0, hard_forward=True):
    y_intersection = x1 * x2
    y_union = x1 + x2 - x1 * x2
    y_symmetric_difference = x1 + x2 - 2.0 * x1 * x2

    # -------- HARD FORWARD --------
    gate = tf.where(
        theta < 0.25,
        0,
        tf.where(theta < 0.75, 1, 2)
    )

    y_hard = tf.where(
        gate == 0,
        y_intersection,
        tf.where(
            gate == 1,
            y_union,
            y_symmetric_difference
        )
    )

    # -------- SOFT FORWARD --------

    positions = tf.constant(
        [0.0, 0.5, 1.0],
        dtype=theta.dtype
    )

    values = tf.stack(
        [y_intersection, y_union, y_symmetric_difference],
        axis=-1
    )

    logits = -tf.square(theta - positions) / temperature
    p = tf.nn.softmax(logits, axis=-1)
    y_soft = tf.reduce_sum(values * p, axis=-1)

    y = tf.where(hard_forward, y_hard, y_soft)

    # -------- BACKWARD --------
    def grad(dy):
        # d logits / d theta
        dz = -2.0 * (theta - positions) / temperature

        # d softmax / d theta
        expected_dz = tf.reduce_sum(p * dz, axis=-1)
        dp = p * (dz - expected_dz[..., None])

        # derivative of surrogate output
        dy_dtheta = tf.reduce_sum(dp * values, axis=-1)

        # gradient wrt theta
        dtheta = tf.reduce_sum(dy * dy_dtheta)

        # Actual derivative wrt inputs for selected operation
        dx1_intersection = x2
        dx1_union = 1.0 - x2
        dx1_symmetric_difference = 1.0 - 2.0 * x2

        dx2_intersection = x1
        dx2_union = 1.0 - x1
        dx2_symmetric_difference = 1.0 - 2.0 * x1

        dx1 = tf.where(
            gate == 0,
            dx1_intersection,
            tf.where(
                gate == 1,
                dx1_union,
                dx1_symmetric_difference
            )
        )

        dx2 = tf.where(
            gate == 0,
            dx2_intersection,
            tf.where(
                gate == 1,
                dx2_union,
                dx2_symmetric_difference
            )
        )

        return (
            dy * dx1,
            dy * dx2,
            dtheta,
            None,
            None
        )

    return y, grad


class PredicateLayer(tf.keras.layers.Layer):

    def __init__(self, temperature=0.1, hard_forward=True, theta_initializer=tf.keras.initializers.Constant(0.5),
                 **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.hard_forward = tf.Variable(hard_forward, dtype=tf.bool, trainable=False, name="hard_forward")
        self.theta_initializer = tf.keras.initializers.get(theta_initializer)

    def build(self, input_shape):
        n = int(input_shape[-1])
        # Number of unique pairs
        self.n_pairs = n * (n - 1) // 2
        # One theta for EVERY pair
        self.theta = self.add_weight(
            name="theta",
            shape=(self.n_pairs,),
            initializer=self.theta_initializer,
            trainable=True
        )
        # Store the pair indices
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        self.pairs = pairs

    def set_hard_forward(self, value):
        self.hard_forward.assign(bool(value))

    def call(self, x):
        outputs = []

        for k, (i, j) in enumerate(self.pairs):
            y = predicate_neuron(
                x[:, i],
                x[:, j],
                self.theta[k],
                self.temperature,
                self.hard_forward)
            outputs.append(y)
        return tf.stack(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        """
        Output shape: (batch, n_pairs) where n_pairs = n * (n - 1) // 2
        """
        n = int(input_shape[-1])
        n_pairs = n * (n - 1) // 2
        return input_shape[:-1] + (n_pairs,)


class MinimalPredicateLayer(tf.keras.layers.Layer):

    def __init__(self, temperature=0.1, hard_forward=True, theta_initializer=tf.keras.initializers.Constant(0.5),
                 **kwargs):
        super().__init__(**kwargs)

        self.temperature = temperature
        self.hard_forward = tf.Variable(hard_forward, dtype=tf.bool, trainable=False, name="hard_forward")
        self.theta_initializer = tf.keras.initializers.get(theta_initializer)

    def build(self, input_shape):
        n = input_shape[-1]

        if n is None:
            raise ValueError("The last input dimension must be known when building PredicateLayer.")

        self.n_inputs = int(n)

        self.n_pairs = self.n_inputs // 2

        self.theta = self.add_weight(
            name="theta",
            shape=(self.n_pairs,),
            initializer=self.theta_initializer,
            trainable=True
        )

        super().build(input_shape)

    def set_hard_forward(self, value):
        self.hard_forward.assign(bool(value))

    def call(self, x):
        outputs = []

        for k in range(self.n_pairs):
            i = 2 * k
            j = i + 1

            y = predicate_neuron(
                x[:, i],
                x[:, j],
                self.theta[k],
                self.temperature,
                self.hard_forward
            )

            outputs.append(y)

        if self.n_inputs % 2 == 1:
            outputs.append(x[:, -1])

        return tf.stack(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        n = int(input_shape[-1])
        n_pairs = n // 2
        output_dim = n_pairs + (1 if n % 2 == 1 else 0)
        return input_shape[:-1] + (output_dim,)
