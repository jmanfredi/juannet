"""
The canonical example of a function that can't be
learned with a simple model is XOR
"""
import numpy as np

from juannet.train import train
from juannet.nn import NeuralNet
from juannet.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
    ])

#I am not sure the outputs below make sense, but Joel
#wants the "true" output to be represented by [0,1]
#and the "false" by [1,0]
targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
    ])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
    ])

train(net, inputs, targets)

for x,y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
