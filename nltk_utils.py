import nltk
import numpy as np
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# tokenization
def tokenize (sentence):
    return nltk.word_tokenize(sentence)

# stemming using PorterStemmer
def stem (word):
    return stemmer.stem(word.lower())

# creating the bag of words
def bag_of_words (toeknized_sentence, all_words):
    # stemming the words in the sentence
    toeknized_sentence = [stem(word) for word in toeknized_sentence]
    # creating a bag of words with 0 values for the length of the all_words list
    bag = np.zeros(len(all_words), dtype=np.float32)
# for each word in the all_words list if each word is in the toeknized_sentence list then make the index of the bag of words value to 1
    for idx, w in enumerate(all_words):
        if w in toeknized_sentence:
            bag[idx] = 1.0
    return bag
   