# Multilayer Perceptron
Multilayer perceptron class written in Python.

## Features

### Allows the user to choose:
* Number of neurons per layer
* Activation function per layer
* Training threshold (Epochs or Convergence Error)
* Type of gradient descent (Stochastic, Batch or Mini-Batch)

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
    
### Optional
#### Initialization
* ```perceptron.learning_rate```

    Learning rate is between 0 and 1, larger values make the weight changes more volatile. Default is 0.01
* ```perceptron.threshold```

    Epoch mode (default):
    ```
    perceptron.threshold = ('epoch', 500) # Train for 500 epochs
    ```
    Convergence error mode:
    ```
    perceptron.threshold = ('error', 0.01) # Train until total error is 0.01 or less (mean-squared error)
    ```
#### Training
* ```perceptron.train(training_inputs, labels, batch_size=(int))```

    Size of training batches. Default value is 1 (Stochastic Gradient Descent).

## Attributes & Methods
* ```perceptron.weights```
    
    List of weights from first to last layer.
* ```perceptron.epochs```
    
    Total number of epochs.
* ```perceptron.predict(training_input)```
    
    Returns the output of the perceptron with its current weights.
