import tensorflow as tf

class SpectralNormCap(tf.keras.constraints.Constraint):
    def __init__(self, cap):
        self.cap = cap
    def __call__(self, w):
        s = tf.linalg.svd(w, compute_uv=False)
        s_max = s[0]
        scale = tf.where(s_max > 0, self.cap / tf.maximum(s_max, self.cap), tf.constant(1.0, dtype=w.dtype))
        return w * scale
    def get_config(self):
        return {"cap": self.cap}
    
class NoisyDense(tf.keras.layers.Dense):
    def __init__(self, units, epsilon=None, delta=None, clip_norm=1.0, spectral_norm_cap=1.0, **kwargs):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.spectral_norm_cap = spectral_norm_cap
        # Ensure kernel spectral norm <= cap at all times
        if epsilon is not None and delta is not None:
            constraint = SpectralNormCap(spectral_norm_cap)
            kwargs["kernel_constraint"] = constraint
        super().__init__(units, **kwargs)

    def call(self, inputs):
        # Optionally clip input norm to bound sensitivity of f(x)=W x
        if self.clip_norm is not None:
            inputs = tf.clip_by_norm(inputs, clip_norm=self.clip_norm, axes=[-1])

        # Linear transform without bias first
        outputs = tf.linalg.matmul(inputs, self.kernel)

        if self.epsilon is not None and self.delta is not None:
            # DP-calibrated Gaussian mechanism
            # Sensitivity S_f = spectral_norm_cap * clip_norm
            input_bound = self.clip_norm if (self.clip_norm is not None) else 1.0
            S_f = tf.cast(self.spectral_norm_cap, outputs.dtype) * tf.cast(input_bound, outputs.dtype)
            # std = S_f * sqrt(2 * log(1.25/delta)) / epsilon
            two = tf.cast(2.0, outputs.dtype)
            one_pt25 = tf.cast(1.25, outputs.dtype)
            eps = tf.cast(self.epsilon, outputs.dtype)
            delt = tf.cast(self.delta, outputs.dtype)
            std = S_f * tf.sqrt(two * tf.math.log(one_pt25 / delt)) / eps
            noise = tf.random.normal(tf.shape(outputs), mean=0.0, stddev=std, dtype=outputs.dtype)
            outputs = outputs + noise
            
        return outputs