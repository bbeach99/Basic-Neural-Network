import numpy


class OptimizerAdam:
    def __init__(self, layers, learning_rate=1, decay=0, epsilon=1e-7, beta_momentum=.9, beta_cache=.999):
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_momentum = beta_momentum
        self.beta_cache = beta_cache
        self.iterations = 0
        self.layers = layers

    def update_layers(self):
        # decay learning rate
        if self.decay:
            self.learning_rate = self.base_learning_rate * (1 / (1 + self.decay * self.iterations))

        for layer in self.layers:
            # calculate new momentums
            layer.weights_momentum = self.beta_momentum * layer.weights_momentum \
                + (1 - self.beta_momentum) * layer.weights_derivative
            layer.biases_momentum = self.beta_momentum * layer.biases_momentum \
                + (1 - self.beta_momentum) * layer.biases_derivative

            # bias correct momentums
            weights_momentum_bias_corrected = layer.weights_momentum / (1 - self.beta_momentum ** (self.iterations + 1))
            biases_momentum_bias_corrected = layer.biases_momentum / (1 - self.beta_momentum ** (self.iterations + 1))

            # calculate new caches
            layer.weights_cache = self.beta_cache * layer.weights_cache \
                + (1 - self.beta_cache) * layer.weights_derivative**2
            layer.biases_cache = self.beta_cache * layer.biases_cache \
                + (1 - self.beta_cache) * layer.biases_derivative ** 2

            # bias correct caches
            weights_cache_bias_corrected = layer.weights_cache / (1 - self.beta_cache ** (self.iterations + 1))
            biases_cache_bias_corrected = layer.biases_cache / (1 - self.beta_cache ** (self.iterations + 1))

            # update weights and biases
            layer.weights += -self.learning_rate * weights_momentum_bias_corrected \
                / (numpy.sqrt(weights_cache_bias_corrected) + self.epsilon)
            layer.biases += -self.learning_rate * biases_momentum_bias_corrected \
                / (numpy.sqrt(biases_cache_bias_corrected) + self.epsilon)

        self.iterations += 1
