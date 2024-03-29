from Layer import Layer
import numpy


class RLActivationLayer(Layer):
    def __init__(self):
        self.input = None

    def calculate_layer(self, inputs):
        self.input = inputs
        return numpy.maximum(0, inputs)

    def back_propagate(self, gradient):
        derivative = gradient.copy()
        derivative[self.input <= 0] = 0
        return derivative
