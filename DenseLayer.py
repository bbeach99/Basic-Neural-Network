from Layer import Layer
import numpy


class DenseLayer(Layer):
    def __init__(self, input_nodes, size):
        self.input = None

        self.weights_derivative = None
        self.biases_derivative = None

        self.weights = 0.01 * numpy.random.randn(input_nodes, size)
        self.biases = [0 for _ in range(size)]

        self.weights_momentum = numpy.zeros_like(self.weights)
        self.biases_momentum = numpy.zeros_like(self.biases)

    def calculate_layer(self, inputs):
        self.input = inputs
        return numpy.dot(inputs, self.weights) + self.biases

    def back_propagate(self, gradient):

        inputs_derivative = numpy.dot(gradient, self.weights.T)
        self.weights_derivative = numpy.dot(self.input.T, gradient)
        self.biases_derivative = numpy.sum(gradient, axis=0, keepdims=True)

        return inputs_derivative

    def apply_gradient(self, learning_rate, momentum):
        # store the gradiant applied to the weights and biases, so it can be used as momentum to calculate
        # the next gradient
        self.weights_momentum = momentum * self.weights_momentum - learning_rate * self.weights_derivative
        self.biases_momentum = momentum * self.biases_momentum - learning_rate * self.biases_derivative

        self.weights += self.weights_momentum
        self.biases += self.biases_momentum
