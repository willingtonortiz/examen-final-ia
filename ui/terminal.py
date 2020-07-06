import time
import csv
import threading
from nlp import nlp
from som import som
from nn.neuralnetwork import NN


class Terminal:
    def __init__(self):
        self.clusters = None
        self.vocabulary_size = 3
        self.agent = NN([self.vocabulary_size, 5, 10, 1])
        self.classify_result = None
        self.vocabulary = None

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

    def nn_classify_sentence_screen(self):
        def execute_classify(sentences):
            tokenized_sentences = nlp.tokenize_sentences(sentences)
            # # NLP -> Generando vectores
            vectors = nlp.generate_vectors(
                tokenized_sentences, self.vocabulary)
            self.classify_result = self.agent.update(vectors[0])

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
        # with open(csv_path) as csvFile:
        #     csvReader = csv.DictReader(csvFile)
        #     for rows in csvReader:
        #         data.append(rows)
        sentences = ['hola antoni', 'chau antoni',
                     'hola antoni', 'chau antoni']
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
                for i, cluster in enumerate(self.clusters):
                    for data in cluster:
                        # Example:  [10,0,0,0,0,0]
                        self.agent.update(data)
                        # Example: 1
                        self.agent.backPropagate(0, i)

                # self.nn_training_screen(train_supervised_nn, csv_path)
            elif selected_option == 3:
                # result = self.agent.update(input)
                self.nn_classify_sentence_screen()
            elif selected_option == 4:
                dont_exit_program = False
