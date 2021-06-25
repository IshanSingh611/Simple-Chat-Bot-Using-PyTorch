import random
import json
import torch
from model import NeuralNetwork
from nltk_utils import bag_of_words,stem,tokenize


device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open("intents.json") as f:
    intents = json.load(f)

FILE_NAME = 'data.pth'
data = torch.load(FILE_NAME)

input_size = data["input_size"]
hidden_size = data['hidden_size']
output_size = data['output_size']
tags = data['tags']
model_state = data['model_state']
all_words = data['all_words']

model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


BOT_NAME = "NIM_BOT"
print("Start conversation ! type 'exit' to stop")
while True:
    user_input = input('You : ->  ')
    if user_input == 'exit':
        break
    user_input = tokenize(user_input)
    X=bag_of_words(user_input,all_words)
    X = X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)
    outputs = model(X)
    _,predicted = torch.max(outputs,dim=1)
    tag=tags[predicted.item()]
    probs=torch.softmax(outputs,dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{BOT_NAME}: -> {random.choice(intent['responses'])}")
    else:
        print(f"{BOT_NAME}: Areyyy kehna kya chahte ho")            




    
    
