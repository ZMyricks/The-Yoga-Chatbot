# import libraries
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np

def tokenize_yoga_data(sentence):
    """
    split sentence into array of tokens
    parameter : sentence
    """
    return nltk.word_tokenize(sentence)


def stem_and_lower(word):
    """
    find the root form of the word
    convert to lowercase
    parameter : word/token
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array

    parameter : returned tokenized sentence, stemmed words
    """
    # stem each word
    sentence_words = [stem_and_lower(word) for word in tokenized_sentence]

    # initialize bag with 0 for each word with NumPy
    theBag = np.zeros(len(words), dtype=np.float32)
    for theBag_index, w in enumerate(words):
        if w in sentence_words:
            theBag[theBag_index] = 1

    return theBag
