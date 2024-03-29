# import random
import numpy


# class to hold all the values of a layer of a NN and calculate its output
# @param input_nodes: number of input values
# @param size: number of nodes in this layer
# @field weighted_inputs: array to store the input * weights + biases
# @field weight_gradient: 2d array to store the sum of the gradient that
# needs to be applied to each weight from each round of back propagation
# @field bias_gradient: array to store the gradient for each bias
# @field weights: 2d array of weights to by multiplied with each input
# @field biases: array to hold the bias of each node
class Layer:

    @staticmethod
    def sigmoid_function(value):
        return 1 / (1 + numpy.exp(-value))

    @staticmethod
    def sigmoid_derivative(value):
        sig = Layer.sigmoid_function(value)
        return sig * (1 - sig)

    # function to calculate the output of the layer
    # @param inputs: input from the previous layer
    # @param output: output of this layer
    def calculate_layer(self, inputs):
        pass

    def back_propagate(self, gradient):
        pass
