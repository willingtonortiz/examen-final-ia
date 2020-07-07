import time
import csv
import threading
from nlp import nlp
from som import som
from nn.neuralnetwork import NN


class Terminal:
    def __init__(self):
        self.clusters = None
        self.vocabulary_size = 1000
        self.agent = NN([self.vocabulary_size, 50, 25, 1], 0.10)
        self.classify_result = None
        self.vocabulary = None
        self.dataset = None

    def clear_screen(self):
        print(chr(27) + "[2J")

    def print_main_menu(self):
        print('┌─────────────────────────────────────────┐')
        print('│           Clasificador de spam          │')
        print('├─────────────────────────────────────────┤')
        print('│ 1. Entrenar red neuronal no supervisada │')
        print('│ 2. Entrenar red neuronal supervisada    │')
        print('│ 3. Clasificar oración                   │')
        print('│ 4. Salir                                │')
        print('└─────────────────────────────────────────┘')

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

    def nn_training_screen(self, neural_network_run, csv_path):
        print(f'La red neuronal entrenará con el archivo {csv_path}')
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

    def run(self, csv_path):
        dont_exit_program = True
        data = []
        with open(csv_path) as csvFile:
            csvReader = csv.DictReader(csvFile)
            i = 0
            for rows in csvReader:
                if i > 5000:
                    break
                data.append(rows)
                i += 1
        self.dataset = data
        sentences = [w["Message"] for w in data]

        tokenized_sentences = nlp.tokenize_sentences(sentences)
        # NLP -> Generando vectores
        self.vocabulary = nlp.build_vocabulary(
            tokenized_sentences, self.vocabulary_size)

        while dont_exit_program:
            self.clear_screen()
            self.print_main_menu()
            selected_option = self.get_input(
                'Ingrese la opción: ', ('1', '2', '3', '4'), int)
            if selected_option == 1:
                def train_som():
                    vectors = nlp.generate_vectors(
                        tokenized_sentences, self.vocabulary)
                    # SOM -> Generando clusters
                    results = som.som(vectors, 1, 2)

                    # Agregar 50 elementos clasificados
                    # results[0].append([1, 1, 0])
                    # results[1].append([0, 0, 1])

                    self.clusters = results

                self.nn_training_screen(train_som, csv_path)

            elif selected_option == 2:
                def train_nn_from_excel():
                    for data in self.dataset:
                        tokenized_sentences = nlp.tokenize_sentences(
                            [data["Message"]])
                        vectors = nlp.generate_vectors(
                            tokenized_sentences, self.vocabulary)
                        self.agent.update(vectors[0])
                        cluster = 0
                        if data["Category"] == "spam":
                            cluster = 1
                        self.agent.backPropagate(0, cluster)

                def train_nn_from_som():
                    for i, cluster in enumerate(self.clusters):
                        for data in cluster:
                            self.agent.update(data)
                            self.agent.backPropagate(0, i)

                self.nn_training_screen(train_nn_from_excel, csv_path)
            elif selected_option == 3:
                def execute_classify(sentences):
                    tokenized_sentences = nlp.tokenize_sentences(sentences)
                    # # NLP -> Generando vectores
                    vectors = nlp.generate_vectors(
                        tokenized_sentences, self.vocabulary)
                    self.classify_result = self.agent.update(vectors[0])
                self.nn_classify_sentence_screen(execute_classify)
            elif selected_option == 4:
                dont_exit_program = False
