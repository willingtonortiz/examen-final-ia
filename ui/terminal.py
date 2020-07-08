import time
import csv
import threading
from nlp import nlp
from som.som import Som
from nn.neuralnetwork import NN
import numpy as np


class Terminal:
    def __init__(self):
        self.clusters = None
        self.vocabulary_size = 1000
        self.agent = NN([self.vocabulary_size, 50, 1], 0.10)
        self.classify_result = None
        self.exam = None
        self.vocabulary = None
        self.dataset = None
        self.names = {}

    def clear_screen(self):
        print(chr(27) + "[2J")

    def print_main_menu(self):
        print('┌────────────────────────────────────────────────────────────────┐')
        print('│                      Clasificador de spam                      │')
        print('├────────────────────────────────────────────────────────────────┤')
        print('│ 1. Entrenar red neuronal no supervisada y etiquetar 1000 datos │')
        print('│ 2. Entrenar red neuronal supervisada                           │')
        print('│ 3. Clasificar oración                                          │')
        print('│ 4. Salir                                                       │')
        print('└────────────────────────────────────────────────────────────────┘')

    def get_input(self, input_message, valid_inputs, data_type):
        incorrect_input = True
        while incorrect_input:
            selected_option_str = input(input_message)
            incorrect_input = not selected_option_str in (valid_inputs)
        selected_option = data_type(selected_option_str)
        return selected_option

    def print_loading_message(self):
        loading_symbols = ['-', '\\', '|', '/']
        for symbol in loading_symbols:
            print('Cargando' + symbol, end='\r')
            time.sleep(0.05)

    def nn_training_screen(self, neural_network_run):
        algorithm_thread = threading.Thread(target=neural_network_run)
        algorithm_thread.start()
        while algorithm_thread.is_alive():
            self.print_loading_message()
        print('Listo    ')
        input('Presione Enter para continuar')

    def nn_classify_sentence_screen(self, execute_classify):
        sentence = input('Ingrese la oración a clasificar: ')
        algorithm_thread = threading.Thread(
            target=execute_classify([sentence]))
        algorithm_thread.start()
        while algorithm_thread.is_alive():
            self.print_loading_message()
        print('Listo    ')
        print(self.classify_result)
        input('Presione Enter para continuar')

    def create_vocabulary(self):
        sentences = self.read_unlabeled_dataset_rows()
        tokenized_sentences = nlp.tokenize_sentences(sentences)
        self.vocabulary = nlp.build_vocabulary(
            tokenized_sentences, self.vocabulary_size)
        print("VOCABULARIO CARGADO!")

    def read_labeled_dataset_rows(self):
        data = []
        with open('labeled_dataset.csv') as csvFile:
            csvReader = csv.DictReader(csvFile)
            i = 1
            for rows in csvReader:
                if i > 50:
                    break
                data.append(rows)
                i += 1
        sentences = [w["Message"] for w in data]
        return sentences

    def read_labeled_dataset_Categories(self):
        data = []
        with open('labeled_dataset.csv') as csvFile:
            csvReader = csv.DictReader(csvFile)
            i = 1
            for rows in csvReader:
                if i > 50:
                    break
                data.append(rows)
                i += 1
        sentences = [w["Category"] for w in data]
        return sentences

    def read_unlabeled_dataset_rows(self):
        data = []
        exam = []
        with open('unlabeled_dataset.csv') as csvFile:
            csvReader = csv.DictReader(csvFile)
            i = 0
            for rows in csvReader:
                if i < 1000:
                    data.append(rows)
                elif i < 150:
                    exam.append(rows)
                else:
                    break
                i += 1
        self.dataset = data
        self.exam = exam
        sentences = [w["Message"] for w in data]
        return sentences

    def nn_label_dataset_rows(self, csv_path, nn_label_run):
        print(f'Clasificando 1000 filas del archivo {csv_path}')
        algorithm_thread = threading.Thread(target=nn_label_run(csv_path))
        algorithm_thread.start()
        while algorithm_thread.is_alive():
            self.print_loading_message()
        print('Listo    ')
        input('Presione Enter para continuar')

    def run(self):

        # ========== Generando vocabulario ========== #
        self.create_vocabulary()
        dont_exit_program = True

        while dont_exit_program:
            self.clear_screen()
            self.print_main_menu()
            selected_option = self.get_input(
                'Ingrese la opción: ', ('1', '2', '3', '4', '5'), int)
            if selected_option == 1:
                def train_som():
                    # ========== FASE DE ENTRENAMIENTO ========== #
                    # Obteniendo dataset de 50 elementos
                    sentences = self.read_labeled_dataset_rows()
                    categories = self.read_labeled_dataset_Categories()
                    tokenized_sentences = nlp.tokenize_sentences(sentences)

                    # NLP -> Generando vectores
                    vectors = nlp.generate_vectors(
                        tokenized_sentences, self.vocabulary)

                    # SOM -> Generando clusters (2 clusters => 50 elementos)
                    data = [{'type': categories[i], 'data': val}
                            for i, val in enumerate(vectors)]

                    som = Som(1, 2)
                    som.train(vectors)
                    clusters = som.test_many(data)

                    # ========== DEFINIENDO SPAM Y NO SPAM ========== #

                    zero_counter = 0
                    one_counter = 0
                    for i, cluster in enumerate(clusters):
                        for item in cluster:
                            if i == 0 and item['type'] == "spam":
                                zero_counter += 1
                            elif i == 1 and item['type'] == "spam":
                                one_counter += 1

                    # Colocando nombres
                    names = {}
                    if zero_counter > one_counter:
                        names[0] = "spam"
                        names[1] = "no spam"
                    else:
                        names[0] = "no spam"
                        names[1] = "spam"
                    self.names = names

                    # Reajustando clusters
                    clusters = [[item['data'] for item in cluster]
                                for cluster in clusters]

                    # ========== FASE DE ETIQUETADO Y AGRUPAMIENTO DE DATASETS ========== #

                    # Vectorizando 1000 datos
                    sentences = self.read_unlabeled_dataset_rows()
                    tokenized_sentences = nlp.tokenize_sentences(sentences)
                    vectors = nlp.generate_vectors(
                        tokenized_sentences, self.vocabulary)

                    # Etiquetando y agrupando
                    for vector in vectors:
                        tag = som.test_one(vector)
                        clusters[tag].append(vector)

                    self.clusters = clusters

                print(f'La red neuronal entrenará con el archivo unlabeled_database.csv')
                self.nn_training_screen(train_som)

            elif selected_option == 2:
                def train_nn_from_excel():
                    """
                    Entrena la red supervisada a partir de los datos etiquetados del excel
                    """
                    epoch = 100
                    errors = []
                    for i in range(epoch):
                        error = 0
                        for data in self.dataset:
                            tokenized_sentences = nlp.tokenize_sentences(
                                [data["Message"]])
                            vectors = nlp.generate_vectors(
                                tokenized_sentences, self.vocabulary)
                            result = self.agent.update(vectors[0])
                            cluster = 0
                            if data["Category"] == "spam":
                                cluster = 1
                            error += pow(cluster-result[0], 2)
                            self.agent.backPropagate(0, cluster)
                        errors.append(error*0.5)

                def train_nn_from_som():
                    """
                    Entrena la red supervisada a partir de los datos etiquetados del la red SOM
                    """
                    epoch = 100
                    errors = []
                    for e in range(epoch):
                        error = 0
                        for i, cluster in enumerate(self.clusters):
                            for data in cluster:
                                result = self.agent.update(data)
                                error += pow(i-result[0], 2)
                                self.agent.backPropagate(0, i)
                        errors.append(error*0.5)

                self.nn_training_screen(train_nn_from_som)

            elif selected_option == 3:
                def execute_classify(sentences):
                    tokenized_sentences = nlp.tokenize_sentences(sentences)
                    # # NLP -> Generando vectores
                    vectors = nlp.generate_vectors(
                        tokenized_sentences, self.vocabulary)
                    self.classify_result = self.names[0] if self.agent.update(
                        vectors[0])[0] > 0.5 else self.names[1]
                self.nn_classify_sentence_screen(execute_classify)

            elif selected_option == 4:
                dont_exit_program = False
