import tensorflow as tf
class NoisyDense(tf.keras.layers.Dense):
    def __init__(self, units, noise_stddev=0.1, **kwargs):
        """
        Adds Gaussian noise N(0, noise_stddev^2) to the pre-activation (W x)
        before bias is added. Noise is applied only when training=True.
        """
        super().__init__(units, **kwargs)
        self.noise_stddev = noise_stddev

    def call(self, inputs):
        # Linear transform without bias first
        outputs = tf.linalg.matmul(inputs, self.kernel)
        
        # Add noise only during training:
        noise = tf.random.normal(
            tf.shape(outputs),
            mean=0.0,
            stddev=self.noise_stddev,
            dtype=outputs.dtype
        )
        outputs = outputs + noise

        # Now add bias (if enabled), then activation
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs