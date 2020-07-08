from ui.terminal import Terminal
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def readCSVFile(csvFilePath, limit=2):
    data = []
    with open(csvFilePath) as csvFile:
        csvReader = csv.DictReader(csvFile)
        i = 0
        for rows in csvReader:
            if i > limit:
                break
            data.append(rows)
            i += 1
    return data

def cleanText(dataset, attribute="Message", language="english"):
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words(language))
    for data in dataset:
        tokens = word_tokenize(data[attribute])
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word for word in tokens if word.isalpha()]
        porter = PorterStemmer()
        tokens = set([porter.stem(word) for word in tokens])
        tokens = [w for w in tokens]
        data[attribute] = tokens
    return dataset


if __name__ == "__main__":
    t = Terminal()
    t.run()

