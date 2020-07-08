import random


class Neuron:
    def __init__(self, total_inputs, activation_function, derivate_activation_function):
        """
        Inicializa la neurona
        """
        self.weights = [random.uniform(-1, 1) for _ in range(total_inputs)]
        self.bias = random.uniform(-1, 1)
        self.activation_function = activation_function
        self.derivate_activation_function = derivate_activation_function
        self.error = 0
        self.result = 0

    def output(self, data_inputs):
        """
        Muestra el resultado de la neurona
        """
        # print("OUTPUT: ")
        weight_data_input_it = zip(self.weights, data_inputs)
        self.result = sum([weight * data_input for weight,
                           data_input in weight_data_input_it]) + self.bias
        # print(self.result)
        self.result = self.activation_function(self.result)
        # print(self.result)
        return self.result

    def calculate_error(self, error):
        """
        Calcula el error de la neurona
        """
        self.error = self.derivate_activation_function(self.result) * error
        return error

    def update_weights(self, learning_rate, data_inputs):
        """
        Actualiza el peso de la neurona
        """
        data_inputs_weights_it = zip(data_inputs, self.weights)
        self.weights = [weight + learning_rate * self.error *
                        data_input for data_input, weight in data_inputs_weights_it]
        self.bias += learning_rate * self.error
