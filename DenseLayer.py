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

        self.weights_cache = numpy.zeros_like(self.weights)
        self.biases_cache = numpy.zeros_like(self.biases)

    def calculate_layer(self, inputs):
        self.input = inputs
        return numpy.dot(inputs, self.weights) + self.biases

    def back_propagate(self, gradient):

        inputs_derivative = numpy.dot(gradient, self.weights.T)
        self.weights_derivative = numpy.dot(self.input.T, gradient)
        self.biases_derivative = numpy.sum(gradient, axis=0, keepdims=True)

        return inputs_derivative
