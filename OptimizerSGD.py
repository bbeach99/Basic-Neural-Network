class OptimizerSGD:
    def __init__(self, layers, learning_rate=1, decay=0, momentum=0):
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
        self.layers = layers

    def update_layers(self):
        # decay learning rate
        self.learning_rate = self.base_learning_rate * (1 / (1 + self.decay * self.iterations))

        # calculate new momentum and update weights and biases for each layer
        for layer in self.layers:
            layer.weights_momentum = self.momentum * layer.weights_momentum - self.learning_rate * layer.weights_derivative
            layer.biases_momentum = self.momentum * layer.biases_momentum - self.learning_rate * layer.biases_derivative

            layer.weights += layer.weights_momentum
            layer.biases += layer.biases_momentum

        self.iterations += 1
