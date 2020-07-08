import random
import math


# Retorna la distancia euclideana entre 2 vectores
def euclidean_distance(a, b):
    size = len(a)
    result = 0

    for i in range(size):
        result += (a[i] - b[i]) ** 2

    return math.sqrt(result)


# Retorna la distancia manharan entre 2 puntos
def manhattan_distance(y1, x1, y2, x2):
    return abs(x1 - x2) + abs(y1 - y2)


# Encuentra la neurona más cercana al vector brindado
# vector: a evaluar
# weights: matriz con las neuronas
# rows, columns: Filas y columnas de la matriz
def nearest_neuron(vector, weights, rows, columns):
    result = (0, 0)
    # Se inicializa la distancia con un valor muy grande
    minimun_distance = 1.0e9

    for i in range(rows):
        for j in range(columns):
            distance = euclidean_distance(weights[i][j], vector)

            # Se asigna la nueva neurona mas cercana a result
            if distance < minimun_distance:
                minimun_distance = distance
                result = (i, j)

    return result


# Ajusta el peso de una neurona en función del alpha y el dato
def adjust_weight(value, alpha, data):
    for i, val in enumerate(value):
        value[i] = value[i] + alpha * (data[i] - value[i])

    return value


# Funcion para ejecutar el SOM
# data ->
class Som:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.weights = []

    def train(self, data):
        data_size = len(data)
        data_dimensions = len(data[0])

        # Inicializando variables
        dimensions = data_dimensions
        iterations = 1000  # variable?

        max_range = self.rows + self.cols
        learning_rate = 0.5  # variable?
        # training_data = data

        # Creando las neuronas
        self.weights = [[[random.random() for _ in range(dimensions)]
                    for _ in range(self.cols)] for _ in range(self.rows)]

        # Ejecutando algoritmo del SOM
        for it in range(iterations):
            alpha = 1.0 - it / iterations
            current_alpha = alpha * learning_rate
            current_range = int(alpha * max_range)

            # Eligiendo un elemento aleatorio del dataset
            t = random.randint(0, data_size - 1)

            (bmu_row, bmu_col) = nearest_neuron(data[t], self.weights, self.rows, self.cols)

            # Ajustando las neuronas de la vecindad
            for i in range(self.rows):
                for j in range(self.cols):
                    if manhattan_distance(bmu_row, bmu_col, i, j) < current_range:
                        self.weights[i][j] = adjust_weight(
                            self.weights[i][j], current_alpha, data[t])

    def test_one(self, item):
        (bmu_row, bmu_col) = nearest_neuron(item, self.weights, self.rows, self.cols)
        return bmu_row * self.cols + bmu_col

    def test_many(self, data):
        # Clasificando los inputs en los clusters generados
        # Neurona = [rows][cols][dimension]
        # Item = [dimension]

        # Clasificando la data en los (rows * cols) clusters
        results = [[] for _ in range(self.rows * self.cols)]
        for i, item in enumerate(data):
            (bmu_row, bmu_col) = nearest_neuron(item, self.weights, self.rows, self.cols)
            results[bmu_row * self.cols + bmu_col].append(item)

        return results
