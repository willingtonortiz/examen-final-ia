from DQN.neuron import Neuron
import math
import numpy as np

# Funciones de activacion
def sigmoid(x):
    return 1/(1+math.e**(-x))

def dsigmoid(y):
    return 1.0*y - y**2

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2


class Layer:
    """
    Representa la capa de una red neuronal
    """
    def __init__(self, inputs_per_neuron, total_neurons):
        """
        Params:\n
        inputs_per_neuron: Cantidad de entradas que tiene cada neurona\n
        total_neurons: Cantidad de neuronas
        """
        self.neurons = [Neuron(inputs_per_neuron, sigmoid, dsigmoid)
                        for _ in range(total_neurons)]

    def predict(self, data_inputs):
        """
        Predicción de datos
        """
        return [neuron.output(data_inputs) for neuron in self.neurons]

    def output(self):
        """
        Representa la salida de la capa
        """
        return [neuron.result for neuron in self.neurons]

    def size(self):
        """
        Cantidad de neuronas 
        """        
        return len(self.neurons)

    def weigths(self):
        """
        Pesos de las neuronas
        """
        i = 0
        for neuron in self.neurons:
            print("neuron#", i, ":")
            print(neuron.weights, "\n")
            i += 1
