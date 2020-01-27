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

# Simple Feed Forward Neural Network
class FeedForwardNetwork:
    # architecture: a list describing the architecture
    # f_activations: a list of the activation functions
    # d_f_activations: a list of the derivatives of the activation functions
    def __init__(self, architecture, f_activations, d_f_activations):
        self.layers = []
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
