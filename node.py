import numpy as np

# General node class to be used for normal neural networks, NEAT, or RNNs
class Node:
    # Has members f_activation: activation function
    #               d_f_activation: derivative of the activation function
    #               bias: bias
    #               activation: the activation of the neuron
    #               input: the input, for back propagation reasons
    #               delta: error signal, also for back propogation
    #               children: a list of tuples containing the other node as well
    #                           as the weight to the node
    def __init__(self, f_activation, d_f_activation):
        self.f_activation = f_activation
        self.d_f_activation = d_f_activation
        self.bias = np.random.random() - 0.5
        self.activation = 0
        self.input = 0
        self.delta = 0
        self.children = []

    # actually create the children. This would go in __init__, but then it would
    # be impossible to create a neuron without already having a neuron
    def create_children(self, children):
        for child in children:
            self.children.append((child, np.random.random() - 0.5))

    # Finishes forward propogation on itself (applying activation function and
    # bias) and calculates its contribution to its children nodes
    def prop(self):
        if self.f_activation is not None:
            self.activation = self.f_activation(self.input + self.bias)
        if len(self.children) > 0:
            for child, weight in self.children:
                child.input += self.activation * weight
