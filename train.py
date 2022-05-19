from cgitb import text
from ftplib import all_errors
import json
from nltk_utils import tokenized, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
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
#creating bag of words from sentences in xy
  words_bag =  bag_of_words(sentence, all_words)
  x_train.append(words_bag)
  label = tags.index(tag)
  y_train.append(label)

print ("y_train: " ,y_train)
x_train = np.array(x_train)
y_train = np.array(y_train)

#create a class with the lis with bag of words(x_train) and indices of tags(y_train)
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#Test data
batch_size = 8
hidden_size=8
output_size=len(tags)
input_size=len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

#check for GPU support(CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model=NeuralNet(input_size,hidden_size,output_size).to(device)

#use cross entropy loss
criterion = nn.CrossEntropyLoss()
#Use adam optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#num_epochs is number of iterations of training
for epoch in range (num_epochs):
    for(words,labels) in train_loader:
        words=words.to(device)
        labels = labels.to(device)
        #converted to float (CrossEntropyLoss needs a float)
        labels=labels.long()

        outputs = model(words)
        #calculate loss with cross entropy
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#epocs(interations) start counting from 0, hence add 1
#print progess every 100 iterations
    if((epoch+1)%100==0):
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
#after training, print final loss
print(f'final loss, loss={loss.item():.4f}')