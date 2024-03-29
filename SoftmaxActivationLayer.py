from Layer import Layer
import numpy


class SoftmaxActivationLayer(Layer):
    def __init__(self):
        self.input = None
        self.output = None
        self.expected_values = None

    def calculate_layer(self, inputs):
        self.input = inputs
        exp_output = numpy.exp(inputs - numpy.max(inputs, axis=1, keepdims=True))
        self.output = exp_output / numpy.sum(exp_output, axis=1, keepdims=True)
        return self.output

    def back_propagate(self, gradient):
        # combined derivative of cross entropy and softmax
        gradient[range(len(gradient)), self.expected_values] -= 1
        gradient = gradient / len(gradient)
        return gradient
