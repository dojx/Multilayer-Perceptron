import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from perceptron import Perceptron


def foo(arr):
    return 2*np.sin(arr[0]) + 3*np.cos(arr[1])


training_inputs = np.random.uniform(-np.pi, np.pi, (100, 2))
labels = [foo(a) for a in training_inputs]
layers = (
    (25, 'SIG'),
    (1, 'LIN')
)

perceptron = Perceptron(2, layers, threshold=('error', 0.01))
perceptron.train(training_inputs, labels, batch_size=1)

# Graphing results
x = np.arange(-np.pi, np.pi, 0.1)
y = np.arange(-np.pi, np.pi, 0.1)
gx, gy = np.meshgrid(x, y)

gz1 = np.empty_like(gx)
gz2 = np.empty_like(gx)

for i in range(len(x)):
    for j in range(len(y)):
        gz1[i, j] = perceptron.predict(np.array([gx[i, j], gy[i, j]]))[0]  # Trained network
        gz2[i, j] = foo(np.array([gx[i, j], gy[i, j]]))  # Real function

fig1 = plt.figure(1).gca(projection='3d')
plt.title("Trained Perceptron")
plt.xlim = [-3, 3]
plt.ylim = [-3, 3]
plt.zlim = [-3, 3]
fig1.plot_surface(gx, gy, gz1, cmap='viridis')

fig2 = plt.figure(2).gca(projection='3d')
plt.title("Original")
plt.xlim = [-3, 3]
plt.ylim = [-3, 3]
plt.zlim = [-3, 3]
fig2.plot_surface(gx, gy, gz2, cmap='viridis')

plt.show()
