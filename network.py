import numpy as np
from node import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return x if x > 0 else 0

def d_relu(x):
    return 1 if x > 0 else 0

def squared_error(y_hat, y):
    return (y_hat - y) * (y_hat - y)

def d_squared_error(y_hat, y):
    return 2 * (y_hat - y)

# Simple Feed Forward Neural Network
class FeedForwardNetwork:
    # architecture: a list describing the architecture
    # f_activations: a list of the activation functions
    # d_f_activations: a list of the derivatives of the activation functions
    def __init__(self, architecture, f_activations, d_f_activations, f_cost, d_f_cost):
        self.layers = []
        self.d_f_cost = d_f_cost
        self.f_cost = f_cost
        # iterate through the architecture, adding lists of neurons of the right
        # length, with the right activation functions
        for i, layer in enumerate(architecture):
            if i > 0:
                self.layers.append([Node(f_activations[i - 1], d_f_activations[i -
                    1]) for _ in range(layer)])
            else:
                self.layers.append([Node(None, None) for _ in range(layer)])
        # actually create the network
        for i in range(len(self.layers) - 1):
            for child in self.layers[i]:
                child.create_children(self.layers[i + 1])

    # forward propagate based on list input inpt
    def prop(self, inpt):
        for i, neuron in enumerate(self.layers[0]):
            neuron.input = inpt[i]
            neuron.prop()
        for i in range(1, len(self.layers)):
            for neuron in self.layers[i]:
                neuron.prop()
        retval = []
        for neuron in self.layers[-1]:
            retval.append(neuron.activation)
        return retval

    # backpropagate with a given training rate
    # also need a list of average derivatives of the cost function with respect
    # to the output nodes
    def backprop(self, training_rate, errors):
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                for j, neuron in enumerate(self.layers[i]):
                    neuron.backprop(True, errors[j], training_rate)
            else:
                for neuron in self.layers[i]:
                    neuron.backprop(False, 0, training_rate)

    # train the network based on various parameters
    #   inputs: a list of inputs to the network
    #   outputs: a corresponding list of outputs
    #   training_rate: the number the gradient gets multiplied by
    #   epochs: the maximum number of epochs
    #   batch_size: how many training samples to evaluate before backpropagating
    def train(self, inputs, outputs, training_rate, epochs, batch_size):
        assert(len(inputs) % batch_size == 0, "Batch size must divide inputs")
        for i in range(epochs):
            d_errors = [0] * len(self.layers[-1])
            error = 0
            for j in range(len(inputs)):
                output = self.prop(inputs[j])
                for k in range(len(d_errors)):
                    d_errors[k] += self.d_f_cost(output[k], outputs[j][k])
                    error += self.f_cost(output[k], outputs[j][k])
                error /= len(d_errors)
                if (j + 1) % batch_size == 0:
                    self.backprop(training_rate, [_ / batch_size for _ in d_errors])
                    d_errors = [0] * len(self.layers[-1])
            error /= len(inputs)
            print("Epoch " + str(i + 1) + ": Error: " + str(error))

