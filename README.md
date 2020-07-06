# Multilayer Perceptron
Multilayer perceptron class written in Python.

## Features

### Allows the user to choose:
* No. of neurons per layer
* Activation function per layer
* Training threshold (Epochs vs. Convergence Error)
* Type of gradient descent (Stochastic, Batch & Mini-Batch)

## Usage
1.  Copy "perceptron.py" into project directory.
2.  Import "Perceptron" class:
    ```
    from perceptron import Perceptron
    ```
3.  Intialize a multilayer perceptron:
    ```
    perceptron = Perceptron(1, layers) # (Input dimension, Layers)
    ```
    The first argument is the length of each input (must be 1-dimensional). The second argument is a tuple containing the number of neurons and the activation function of every layer. Example:
    ```    
    layers = (
        (15, 'SIG'), # 15 neurons, sigmoid activation
        (1, 'LIN') # 1 neuron, linear activation
    )
    ```
    Here the perceptron will have two layers:
    * First layer will have 15 neurons, all using the sigmoid activation function
    * Last layer will have 1 neuron using the linear activation function
    
 4. Train the perceptron:
    ```
    perceptron.train(training_inputs, labels)
    ```
    Both arguments must be numpy arrays. The length of each label and the no. of neurons in the last layer must be equal.
    
### Optional arguments
