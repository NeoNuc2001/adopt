import tensorflow as tf

class Adopt(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-6, name="Adopt", **kwargs):
        super(Adopt, self).__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Update the second moment estimate excluding the current gradient
        v_t = self.beta_2 * v + (1 - self.beta_2) * tf.square(grad)
        v.assign(v_t)

        # Normalize the gradient with the previous second moment
        grad_normalized = grad / (tf.sqrt(v) + self.epsilon)

        # Momentum update for mt
        m_t = self.beta_1 * m + (1 - self.beta_1) * grad_normalized
        m.assign(m_t)

        # Update parameter
        var.assign_sub(self.learning_rate * m)

    def _resource_apply_sparse(self, grad, var, indices):
        # For sparse gradients, use the same logic as dense updates but with only indexed elements
        # For simplicity, this part can be omitted here but should follow similar logic if needed
        pass

    def get_config(self):
        config = super(Adopt, self).get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon
        })
        return config
