# Multilayer Perceptron
Multilayer perceptron class written in Python.

## Features

### Allows the user to choose:
* Number of neurons per layer
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
    perceptron = Perceptron({input_count}, {layers})
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
    
    By default if this argument is not passed, the perceptron will have 1 layer with 1 neuron using the linear activation function.
    
 4. Train the perceptron:
    ```
    perceptron.train(training_inputs, labels)
    ```
    Both arguments must be numpy arrays. The length of each label and the number of neurons in the last layer must be equal.
    
## Optional attributes
* ```perceptron.threshold = ('epochs', 500) # Train for 500 epochs (default)```:

    Training threshold. Can be switched to convergence error mode:
    ```
    perceptron.threshold = ('error', 0.01) # Train until convergence error is 0.01 or less
    ```
    

