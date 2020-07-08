import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

# Tokeniza un texto de entrada 
# text -> string
# returns -> string[]
def split_text(text):
    # return text.split(" ")
    return word_tokenize(text)


# Filtra palabras indeseadas en el texto
# text -> string[]
# returns -> string[]
def filter_text(text):

    stop_words = set(stopwords.words('english'))
    text = [word for word in text if not word in stop_words]
    text = [word for word in text if word.isalpha()]
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text]

    filters = [".", "_", ","]
    return [word for word in text if word not in filters]


# Divide y filtra un texto
# text -> string
# returns -> string[]
def split_and_filter(text): 
    splitted_text = split_text(text)
    filtered_text = filter_text(splitted_text)
    return filtered_text


# Tokeniza un arreglo de textos
# sentences -> string[]
# returns -> string[][]
def tokenize_sentences(sentences):
    # Transformar a minusculas
    sentences = [sentence.lower() for sentence in sentences]
    # Se encuntran y filtran las palabras
    result = [split_and_filter(sentence) for sentence in sentences]

    return result


# Construye un vocabulario de tamaño 'size' a partir delas oraciones
# sentences -> string[][]
# returns -> set<string>
def build_vocabulary(sentences, size):
    words_map = {}
    for sentence in sentences:

        for word in sentence:
            if words_map.get(word) is None:
                words_map[word] = 1
            else:
                words_map[word] += 1

    counted_words = [(words_map[key], key) for key in words_map.keys()]
    counted_words = sorted(counted_words, reverse=True)

    # Se encuentran las 'size' palabras mas repetidas
    # print(counted_words)
    result = set()
    for i in range(size):
        result.add(counted_words[i][1])

    return result


# Genera vectores de caracteristicas a partir de un vocabulario y un conjunto de oraciones
def generate_vectors(sentences, vocabulary):
    vocabulary_size = len(vocabulary)

    vectors = []
    for sentence in sentences:
        vector = [0 for _ in range(vocabulary_size)]

        for word in sentence:
            for i, w in enumerate(vocabulary):
                if word == w:
                    vector[i] += 1

        vectors.append(vector)

    return vectors


# Funcion main para probar las funciones
if __name__ == '__main__':
    sentences = ["A A A", "B B B B B", "C C C", "D D D D"]

    tokenized_sentences = tokenize_sentences(sentences)
    # print(tokenized_sentences)

    vocabulary = build_vocabulary(tokenized_sentences, 3)
    print(vocabulary)

    vectors = generate_vectors(tokenized_sentences, vocabulary)
    for vector in vectors:
        print(vector)
