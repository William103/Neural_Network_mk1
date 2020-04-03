# Neural_Network_mk1

This is my first successful attempt at building a neural network from scratch.
Me and Alex Xuan (silvernx on Github) worked together on this. There are three
versions of the network in this repo: a singlethreaded version, a multithreaded
version, and a version learning the MNIST dataset.

## Disclaimer
This program is really slow. Part of that is python and the limitations of the
language, part of it is the fact that we built it from scratch (numpy doesn't
count), and part of it is the way we implemented. We decided to implement it
as a graph with nodes rather than with matrices, since it would be more
extensible and simpler that way. 

## How to use
In an effort to combat the slowness, we use manager.py to train a few networks
for a short period of time then choose the best one to train further. The
function `manager.train_nets` is where that magic happens. This function takes a
bunch of arguments:
 - `inputs`: a list of lists of floats consisting of the
training inputs; `outputs`, a corresponding list of lists of floats consisting
of the expected outputs.
 - `training_rate`: the value the gradient gets multiplied
by during gradient descent; `epochs`, the number of epochs to train for.
 - `batch_size`: the number of inputs to go through before updating the network,
smaller values tend to yield better results since it updates the network more,
but making it too small can yield bad results.
 - `outer_min`: this poorly named
variable is the accuracy threshold for the potential networks.
 - `random_limit`: the weights and biases are chosen uniformly from the
interval `[-random_limit/2, random_limit/2]`, larger values than one might
expect tend to work better, since it's so slow and we train so many networks
it's worth the risk of being very far off with tiny gradients.
 -  `layers`: a list enumerating the architecture of the network, so for
example a network with an input layer with 5 inputs, two hidden layers with 16
neurons each, and an output layer with 2 neurons would have a `layers` list of
`[5, 16, 16, 2]`.
- `activations`: a list of the activation functions for each layer, available
functions are `sigmoid` and `relu`, but this is very easy to extend.
 - `d_activations`: the derivatives of the activation functions, just
copy the previous list but prepend d\_, so the options are `d_sigmoid` and
`d_relu`.
 - `cost`: the cost function, just `squared_error` for now.
 - `d_cost`: the derivative of the cost function, just `d_squared_error`.
 - `num_nets`: the number of preliminary nets to train

This function will return the final network, which you can then train by calling
its `train` member function, which just takes `inputs`, `outputs`,
`training_rate`, `epochs`, `batch_size`, and `verbose`. All but the last are the
same as above. Pass `True` to `verbose` for extra diagnostic printing and
`False` otherwise.
