import numpy as np
from node import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def itself(x):
    return x

def d_itself(x):
    return 1

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
    def __init__(self, architecture, f_activations, d_f_activations, f_cost,
            d_f_cost, random_limit):
        self.layers = []
        self.d_f_cost = d_f_cost
        self.f_cost = f_cost
        # iterate through the architecture, adding lists of neurons of the right
        # length, with the right activation functions
        for i, layer in enumerate(architecture):
            if i > 0:
                self.layers.append([Node(f_activations[i - 1], d_f_activations[i -
                    1], random_limit) for _ in range(layer)])
            else:
                self.layers.append([Node(None, None, random_limit) for _ in range(layer)])
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

    # propagate forwards and backwards
    def prop_to_and_fro(self, x, y, training_rate):
        y_hat = self.prop(x)
        self.backprop(y, y_hat, training_rate)
        error = 0
        for i in range(len(y_hat)):
            error += self.f_cost(y_hat[i], y[i])
        error /= len(y_hat)
        return y_hat, error

    # backpropagate with a given training rate
    # to the output nodes
    def backprop(self, y, y_hat, training_rate):
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                for j, neuron in enumerate(self.layers[i]):
                    neuron.backprop(True, self.d_f_cost(y_hat[j], y[j]), training_rate)
            else:
                for neuron in self.layers[i]:
                    neuron.backprop(False, 0, training_rate)
        for layer in self.layers:
            for neuron in layer:
                neuron.delta = 0

    # apply the changes based on data gathered during backpropagation
    def update(self, training_rate, batch_size):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.update(batch_size)

    # train the network based on various parameters
    #   inputs: a list of inputs to the network
    #   outputs: a corresponding list of outputs
    #   training_rate: the number the gradient gets multiplied by
    #   epochs: the maximum number of epochs
    #   batch_size: how many training samples to evaluate before backpropagating
    def train(self, inputs, outputs, training_rate, epochs, batch_size, verbose):
        assert len(inputs) % batch_size == 0, "Batch size must divide inputs"
        for i in range(epochs):
            total_error = 0
            for j in range(len(inputs)):
                output, error = self.prop_to_and_fro(inputs[j], outputs[j], training_rate)
                if verbose and (i - 1) % 100 == 0:
                    print(inputs[j], output)
                total_error += error
                if (j + 1) % batch_size == 0:
                    self.update(training_rate, batch_size)
            total_error /= len(inputs)
            if verbose and i % 100 == 0:
                print("Epoch " + str(i) + ": Error: " + str(total_error))
        return total_error

    # train the network to a certain level of accuracy
    #   training_data:      list of tuples of inputs and corresponding outputs
    #   evaluation_data:    same but for the purpose of evaluating the network
    #   training_rate:      the rate at which the network trains
    #   min_error:          the target error
    #   batch_size:         the amount of samples to do before each evaluation
    #   verbose:            verbose output or no
    def train_to_accuracy(self, training_data, evaluation_data, training_rate, min_error, batch_size, verbose):
        assert min_error > 0 and min_error < 1, "min_accuracy must be between 0 and 1"
        error = 100
        epochs = 0
        index = 0
        batch = 0
        while error > min_error:
            for i in range(batch_size):
                output, _ = self.prop_to_and_fro(training_data[index][0], training_data[index][1], training_rate)
                index += 1
                if index == len(training_data):
                    index = 0
                    epochs += 1
                    if verbose: print('Evaluating network')
                    total_error = 0
                    for i in range(len(evaluation_data[0])):
                        _, error = self.prop_to_and_fro(evaluation_data[i][0], evaluation_data[i][1], 0)
                        total_error += error
                    print('Epoch: ' + str(epochs) + ' error: ' + str(error))
            batch += 1
            self.update(training_rate, batch_size)
            if verbose: print('Done with batch #', batch)
        if verbose: print('Minimum error achieved')

    # clear the networks activations just in case you don't backprop
    def clear(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.input = 0
                neuron.activation = 0
                neuron.deltabias = 0
                for _, weight in neuron.children:
                    weight.deltaweight = 0
