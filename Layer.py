# import random
import numpy


# Parent class of all layer classes
class Layer:

    @staticmethod
    def sigmoid_function(value):
        return 1 / (1 + numpy.exp(-value))

    @staticmethod
    def sigmoid_derivative(value):
        sig = Layer.sigmoid_function(value)
        return sig * (1 - sig)

    def calculate_layer(self, inputs):
        pass

    def back_propagate(self, gradient):
        pass
