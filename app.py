from flask import Flask,render_template,request
import tflearn
import numpy
import nltk
nltk.download('punkt') 
import json
import tensorflow 
import pickle
import random
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()


#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__) 

# model

net = tflearn.input_data(shape=[None, 46])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,6, activation="softmax") 
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.mdl")

with open("data.pickle","rb") as f:
    words, labels, training, output = pickle.load(f)

with open("intents.json") as file:
    data=json.load(file)
"""
english_bot = ChatBot("Chatterbot",storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")
trainer.train("data/data.yml")

def get_bot_response():
     userText = request.args.get("msg") #get data from input,we write js  to index.html
     return str(english_bot.get_response(userText))
"""
def bag_of_words(sentence,words):

     bag = [0 for _ in range(len(words))]

     sentence_words = nltk.word_tokenize(sentence)
     sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

     for se in sentence_words:
          for i,w in enumerate(words):
               if w == se:
                    bag[i] = 1

     return numpy.array(bag)


@app.route("/")
def index():
     return render_template("index.html") #to send context to html

@app.route("/get")
def get_bot_response():
     inp = request.args.get("msg")

     #if inp.lower() == "quit":
      #break

     results = model.predict([bag_of_words(inp,words)])

     results_index = numpy.argmax(results)

     tag = labels[results_index]

     for tg in data["intents"]:
       if tg["tag"] == tag:
         responses = tg["responses"]

     resp = random.choice(responses)

     return str(resp)





if __name__ == "__main__":
     app.run(debug = True)


