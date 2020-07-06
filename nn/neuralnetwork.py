import math
import random
import string
import numpy as np
from nn.layer import Layer


class NN:
    def __init__(self, nodes, learning_rate = 0.01):
        # Layers
        self.hiddenLayers = []

        self.learningRate = learning_rate

        for i in range(1, len(nodes)-1):
            self.hiddenLayers.append(Layer(nodes[i-1], nodes[i]))

        self.outputLayer = Layer(nodes[-2], nodes[-1])

    def update(self, input):
        self.input = input
        self.hiddenLayers[0].predict(input)
        for i in range(1, len(self.hiddenLayers)):
            self.hiddenLayers[i].predict(self.hiddenLayers[i-1].output())
        return self.outputLayer.predict(self.hiddenLayers[-1].output())

    def backPropagate(self, action, value):
        # Calcula el error para la capa de salida
        error = value-self.outputLayer.output()[action]
        self.outputLayer.neurons[action].calculate_error(error)
        
        # Calcula el error para la capa oculta
        for i in range(self.hiddenLayers[-1].size()):
            error = self.outputLayer.neurons[action].error * \
                self.outputLayer.neurons[action].weights[i]
            self.hiddenLayers[-1].neurons[i].calculate_error(error)

        for i in range(len(self.hiddenLayers)-2, -1, -1):
            for j in range(self.hiddenLayers[i].size()):
                error = 0.0
                for k in range(self.hiddenLayers[i+1].size()):
                    error = error + \
                        self.hiddenLayers[i+1].neurons[k].error * \
                        self.hiddenLayers[i+1].neurons[k].weights[j]
                self.hiddenLayers[i].neurons[j].calculate_error(error)
                # hidden_deltas[i][j] = dsigmoid(self.ah[i][k]) * error

        for i in range(1, len(self.hiddenLayers)):
            for neuron in self.hiddenLayers[i].neurons:
                neuron.update_weights(self.learningRate, self.hiddenLayers[i-1].output())
        for neuron in self.hiddenLayers[0].neurons:
            neuron.update_weights(self.learningRate, self.input)

    def weights(self):
        print('Pesos de Entrada:')
        self.hiddenLayers[0].weigths()
        print()
        for i in range(1, len(self.hiddenLayers)):
            print('Peso de capa oculta:')
            self.hiddenLayers[i].weigths()
            print()
        print('Pesos de Salida:')
        self.outputLayer.weigths()

    def output(self):
        return self.outputLayer.output()

    def max(self):
        action = 0
        for i in range(1, self.outputLayer.size()):
            if(self.output()[i] > self.output()[action]):
                action = i
        return action
