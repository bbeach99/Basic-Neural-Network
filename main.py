import numpy
from nnfs.datasets import spiral_data
from keras.datasets import mnist

import DenseLayer
import RLActivationLayer
import SoftmaxActivationLayer
import NeuralNetwork
import OptimizerSGD
import OptimizerAdam


def spiral_NN():
    X, y = spiral_data(samples=1000, classes=3)
    text_X, test_y = spiral_data(samples=100, classes=3)

    L = DenseLayer.DenseLayer(2, 100)
    LA = RLActivationLayer.RLActivationLayer()
    L2 = DenseLayer.DenseLayer(100, 100)
    L2A = RLActivationLayer.RLActivationLayer()
    L3 = DenseLayer.DenseLayer(100, 3)
    L3A = SoftmaxActivationLayer.SoftmaxActivationLayer()
    NN = NeuralNetwork.NeuralNetwork([L, LA, L2, L2A, L3, L3A])
    optimizer = OptimizerAdam.OptimizerAdam([L, L2], .1, 1e-5)

    for i in range(1000):

        output = NN.calculate_nn(X)
        NN.back_propagation(output, y)
        optimizer.update_layers()

        output = NN.calculate_nn(text_X)
        cost = NeuralNetwork.cross_entropy_cost(output, test_y)
        print("Epoch ", i)
        print(" Loss: ", cost)
        answers = NeuralNetwork.answer(output)
        print(" Accuracy: ", numpy.mean(answers == test_y))


def MNIST_NN():
    # Load MNIST data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Convert data set to 2d array
    X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    X_test = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))

    # adjust all values to be between -1 and 1
    X = (X - 127.5) / 255
    X_test = (X_test - 127.5) / 255

    L = DenseLayer.DenseLayer(784, 1000)
    LA = RLActivationLayer.RLActivationLayer()
    L2 = DenseLayer.DenseLayer(1000, 1000)
    L2A = RLActivationLayer.RLActivationLayer()
    L3 = DenseLayer.DenseLayer(1000, 10)
    L3A = SoftmaxActivationLayer.SoftmaxActivationLayer()
    NN = NeuralNetwork.NeuralNetwork([L, LA, L2, L2A, L3, L3A])
    optimizer = OptimizerSGD.OptimizerSGD([L, L2], .1, .0001)

    for i in range(10000):

        output = NN.calculate_nn(X)
        NN.back_propagation(output, train_y)
        optimizer.update_layers()

        # test
        output = NN.calculate_nn(X_test)
        cost = NeuralNetwork.cross_entropy_cost(output, test_y)
        print("Loss: ", cost)
        answers = NeuralNetwork.answer(output)
        print("Accuracy: ", numpy.mean(answers == test_y))


# spiral_NN()
MNIST_NN()
