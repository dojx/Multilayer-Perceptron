import numpy as np

class Perceptron():
    def __init__(self, input_count=1, layers=((1,'LIN'),), threshold=('epochs', 500), learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights, self.activations = self.init_layers(input_count, layers)
        self.error_total = 999999
        self.epochs = 0

    def predict(self, inputs):
        input_list = []
        summation_list = []
        for weights, activation in zip(self.weights, self.activations):
            summation = np.dot(np.squeeze(weights[:, 1:]), np.squeeze(inputs)) + weights[:, 0]
            prediction = activation_functions[activation](summation)
            input_list.append(inputs)
            summation_list.append(summation)
            inputs = prediction
        return prediction, summation_list, input_list

    def train(self, training_inputs, labels, batch_size=1):
        gradients_avg_w = []
        gradients_avg_b = []
        while not self.finished_training():
            error = 0
            offset = (len(training_inputs) % batch_size) * (self.epochs - 1)
            for i, (inputs, label) in enumerate(zip(training_inputs, labels)):
                prediction, summation_list, input_list = self.predict(inputs)
                error += 0.5*(label - prediction)**2
                gradients = self.get_gradients(label, prediction, summation_list)
                if (i + offset) % batch_size == 0:
                    for j in range(len(gradients)):
                        gradients_avg_w.append((self.learning_rate * gradients[j] * input_list[j])/batch_size)
                        gradients_avg_b.append((self.learning_rate * np.squeeze(gradients[j]))/batch_size)
                else:
                    for j in range(len(gradients)):
                        gradients_avg_w[j] += (self.learning_rate * gradients[j] * input_list[j])/batch_size
                        gradients_avg_b[j] += (self.learning_rate * np.squeeze(gradients[j]))/batch_size
                if (i + offset + 1) % batch_size == 0:
                    for j, (gradient_w, gradient_b) in enumerate(zip(gradients_avg_w, gradients_avg_b)):
                        self.weights[j][:, 1:] += gradient_w
                        self.weights[j][:, 0] += gradient_b
                    gradients_avg_w.clear()
                    gradients_avg_b.clear()
            self.error_total = error/len(training_inputs)
            print(self.error_total)

    def get_gradients(self, label, prediction, summation_list):
        gradients = [np.vstack(derived_functions[self.activations[-1]](summation_list[-1]) * (label - prediction))]
        for i, activation in reversed(list(enumerate(self.activations[:-1]))):
            derivative = np.squeeze(np.diag(derived_functions[activation](summation_list[i])))
            weights = np.transpose(np.atleast_2d(self.weights[i + 1][:, 1:]))
            gradients.append(np.dot(np.dot(derivative, weights), gradients[-1]))
        return list(reversed(gradients))

    def init_layers(self, input_count, layers):
        weights = []
        activations = []
        for neuron_count, activation in layers:
            weights.append(np.random.uniform(-1, 1, (neuron_count, input_count + 1)))
            activations.append(activation)
            input_count = neuron_count
        return weights, activations

    def finished_training(self):
        self.epochs += 1
        if self.threshold[0] == 'epochs':
            return self.epochs >= self.threshold[1]
        return self.error_total <= self.threshold[1]

# Activation Functions
def linear(summation):
    return summation

def linear_d(summation):
    return np.array([1])

def sigmoid(summation):
    return 1/(1+np.exp(-summation))

def sigmoid_d(summation):
    return np.exp(-summation)/((1+np.exp(-summation))**2)

def binary_step(summation):
    return np.where(summation >= 0, 1, 0)

def negative_binary_step(summation):
    return np.where(summation >= 0, 1, -1)

def max_binary(summation):
    return np.where(summation == np.max(summation), 1, 0)

activation_functions = {
    'LIN': linear,
    'SIG': sigmoid,
    'STEP': binary_step,
    'NSTEP': negative_binary_step,
    'MAX': max_binary
}

derived_functions = {
    'LIN': linear_d,
    'SIG': sigmoid_d,
    'STEP': linear_d,
    'NSTEP': linear_d,
    'MAX': linear_d
}