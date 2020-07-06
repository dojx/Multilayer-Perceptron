# Multilayer Perceptron
Multilayer perceptron class written in Python.

## Features

### Allows the user to choose:
* No. of neurons per layer
* Activation function per layer
* Training threshold (Epochs vs. Convergence Error)
* Type of gradient descent (Stochastic, Batch & Mini-Batch)

## Setup
1.  Copy "perceptron.py" into project directory.
2.  Import "Perceptron" class:
    ```
    from perceptron import Perceptron
    ```
3.  Intialize a multilayer perceptron:
    ```
    perceptron = Perceptron(1, layers) # (Input dimension, Layers)
    ```
    The first argument is the length of each input. The second argument is the number of neurons and the activation function of every layer. Example:
    ```
    perceptron = Perceptron(1, layers) # (Input dimension, Layers)
    ```
    layers = (
        (15, 'SIG'), # 15 neurons, sigmoid activation
        (1, 'LIN') # 1 neuron, linear activation
    )
    ```
    Here the perceptron will have two layers:
    * First layer will have 15 neurons, all with the sigmoid activation function
    * Seconde layer will have 1 neuron with the linear activation function
