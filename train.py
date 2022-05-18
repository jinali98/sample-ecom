from ftplib import all_errors
import json
from nltk_utils import tokenized, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# this is the training file

# loading the json file 
with open('intent.json', 'r') as f:
    intents = json.load(f)

# this is all the words that are used in the training file
all_words = []
# this is the tags we have in the training file
tags = []
# this will hold the tags with thier corresponding patterns 
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize the pattern and this gives an array of words
        words = tokenized(pattern)
        # as words is an array we extend it and  add the words to the all_words list 
        all_words.extend(words)
        # we add the tag and the corresponding words to the xy  as a tuple
        xy.append((words, tag))


# list of words to ignore
ignore = ['?', '.', '!', ',', ':', ';', '-', '_', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '{', '}', '[', ']', '|', '\\', '"', '\'', '<', '>', '~', '`', ]

# stem the words in the all_words list while removing the ignore words
all_words = [stem(word) for word in all_words if word not in ignore]
# remove duplicates from the all_words list and sort the list
all_words = sorted((set(all_words)))
tags = sorted(set(tags))
print(tags)
# bag of words for each sentence
x_train = []
# associated numbers with the tags
y_train = []

for (sentence, tag) in xy:
  words_bag =  bag_of_words(sentence, all_words)
  x_train.append(words_bag)
  label = tags.index(tag)
  y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
