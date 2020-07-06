import time
import threading


class Terminal:
    def clean_screen(self):
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

    def nn_training_screen(self, neural_network, csv_path):
        def hola():
            for _ in range(1_000_000_000):
                pass

        print(f'La red neuronal entrenará con el archivo {csv_path}')
        algorithm_thread = threading.Thread(target=hola)
        algorithm_thread.start()
        while algorithm_thread.is_alive():
            self.print_loading_message()
        print('Listo    ')
        input('Presione Enter para continuar')

    def nn_classify_sentence_screen(self):
        def hola():
            for _ in range(1_000_000_000):
                pass

        sentence = input('Ingrese la oración a clasificar: ')
        # vectorize sentence
        algorithm_thread = threading.Thread(target=hola)
        algorithm_thread.start()
        while algorithm_thread.is_alive():
            self.print_loading_message()
        print('Listo    ')
        print('Datos...')
        input('Presione Enter para continuar')

    def run(self, csv_path):
        dont_exit_program = True
        while dont_exit_program:
            self.clean_screen()
            self.print_main_menu()
            selected_option = self.get_input(
                'Ingrese la opción: ', ('1', '2', '3', '4'), int)
            if selected_option == 1:
                self.nn_training_screen('not supervised', csv_path)
            elif selected_option == 2:
                self.nn_training_screen('supervised', csv_path)
            elif selected_option == 3:
                self.nn_classify_sentence_screen()
            elif selected_option == 4:
                dont_exit_program = False
