import numpy
from nnfs.datasets import spiral_data

import DenseLayer
import RLActivationLayer
import SoftmaxActivationLayer
import NeuralNetwork


def spiral_NN(base_learning_rate, rate_decay):

    X, y = spiral_data(samples=300, classes=3)

    L = DenseLayer.DenseLayer(2, 100)
    LA = RLActivationLayer.RLActivationLayer()
    L2 = DenseLayer.DenseLayer(100, 3)
    L2A = SoftmaxActivationLayer.SoftmaxActivationLayer()
    NN = NeuralNetwork.NeuralNetwork([L, LA, L2, L2A])

    for i in range(10000):

        learning_rate = base_learning_rate * (1 / (1 + rate_decay * i))

        output = NN.calculate_nn(X)
        cost = NeuralNetwork.cross_entropy_cost(output, y)
        print("Loss: ", cost)
        answers = NeuralNetwork.answer(output)
        print("Accuracy: ", numpy.mean(answers == y))

        L2A.expected_values = y
        NN.back_propagation(output, y)

        L.apply_gradient(learning_rate)
        L2.apply_gradient(learning_rate)


spiral_NN(1, .00001)
