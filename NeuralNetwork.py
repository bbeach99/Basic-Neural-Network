import numpy


def cross_entropy_cost(output, expected_value):
    output_clipped = numpy.clip(output, 1e-7, 1 - 1e-7)
    loss = -numpy.log(output_clipped[range(len(output_clipped)), expected_value])
    return numpy.mean(loss)


def cross_entropy_cost_derivative(output, expected_value):
    expected_value = numpy.eye(len(output[0]))[expected_value]
    derivative = -expected_value / output
    return derivative / len(expected_value)


# returns the index of the array with the highest value
def answer(output):
    results = numpy.argmax(output, axis=1)
    return results


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.output = []

    def calculate_nn(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_layer(inputs)
        return inputs

    def back_propagation(self, gradient, expected_values):
        # simplified chain rule of softmax and cross entropy loss combined
        # gradient = numpy.array(outputs)
        # gradient[range(len(outputs)), expected_values] -= 1
        # gradient = gradient / len(outputs)
        # print(f"\nCross entropy and Softmax D:\n{gradient}")
        # cross_entropy_cost_derivative(gradient, expected_values)
        for layer in reversed(self.layers):
            gradient = layer.back_propagate(gradient)
