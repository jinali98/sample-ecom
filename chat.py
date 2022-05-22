import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# check for GPU support(CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Open intents JSON file
with open('intent.json', 'r') as f:
    intents = json.load(f)

#Open saved data file from training
FILE = 'data.pth'
data = torch.load(FILE)

#Get data from data file
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Implmeting chat for testing

bot_name = "Dilmah"

def get_response(msg):        
        sentence = tokenize(msg)
        x = bag_of_words(sentence, all_words)
        # print(f"SHape of x is {x.shape[0]}")

        #reshaping array
        x = x.reshape(1, x.shape[0])
        # print(f"SHape after reshape : {x.shape[0]}")
        x = torch.from_numpy(x)
        output=model(x)
        _,predicted=torch.max(output,dim=1)
        tag=tags[predicted.item()]
        probs=torch.softmax(output,dim=1)
        prob=probs[0][predicted.item()]

    #if the sentence pattern is simillar to a known pattern, give a realted output
        if prob.item()>0.75:

            for intent in intents["intents"]:
                if tag == intent["tag"]:
                #Output a random choice with same tag
                    return (f"{random.choice(intent['responses'])}")  
    #If sentence pattern is unrecognized, say not understood
        
        return ("Sorry I don't get you!!")
          


