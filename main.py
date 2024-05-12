import numpy
from nnfs.datasets import spiral_data

import DenseLayer
import RLActivationLayer
import SoftmaxActivationLayer
import NeuralNetwork
import OptimizerSGD
import OptimizerAdam


def spiral_NN():

    X, y = spiral_data(samples=300, classes=3)

    L = DenseLayer.DenseLayer(2, 100)
    LA = RLActivationLayer.RLActivationLayer()
    L2 = DenseLayer.DenseLayer(100, 3)
    L2A = SoftmaxActivationLayer.SoftmaxActivationLayer()
    NN = NeuralNetwork.NeuralNetwork([L, LA, L2, L2A])
    # optimizer = OptimizerSGD.OptimizerSGD([L, L2], base_learning_rate, rate_decay, momentum)
    optimizer = OptimizerAdam.OptimizerAdam([L, L2], .1, 1e-7)

    for i in range(10000):

        output = NN.calculate_nn(X)
        if not i % 100:
            cost = NeuralNetwork.cross_entropy_cost(output, y)
            print("Loss: ", cost)
            answers = NeuralNetwork.answer(output)
            print("Accuracy: ", numpy.mean(answers == y))

        NN.back_propagation(output, y)

        optimizer.update_layers()


spiral_NN()
