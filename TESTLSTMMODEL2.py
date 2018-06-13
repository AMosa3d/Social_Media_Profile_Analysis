from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import csv
from keras.preprocessing.sequence import pad_sequences
import pickle

def LoadData():
    TrainingSentences = []
    TrainingLabels = []
    with open('Sentiment Analysis Dataset.csv', 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            TrainingSentences.append(row[3])
            TrainingLabels.append(row[1])


    TrainingSentences = TrainingSentences[1:]
    TrainingLabels = [int(x) for x in TrainingLabels[1:]]
    return TrainingSentences, TrainingLabels

def main(tweets):
 
 # load model from single file


 model_Lstm = load_model('Pos_Neg.h5')
 

 with open('Pos_Neg_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

 X1 = tweets
 TrainingSentencesSequences = tokenizer.texts_to_sequences(X1)
 TrainingSentencesSequences = pad_sequences(TrainingSentencesSequences, maxlen=25)
 yhat1 = model_Lstm.predict_classes(TrainingSentencesSequences)
 return yhat1

'''
 # make predictions
 X1 =["I Love Eating Apple",
      "I Hate Eating Apple",
      "i am so sad for my ex",
      "i love sitting with myself",
      "you don't have a heart",
      "What is your worst Pokemon?",
      "What is your worst memory?",
      "work is sweet pain",
      "i am human",
      "i am soooo happpy",
      "i am soooo saaaad",
      "i am soooo saaaadddd",
      "What's your name?",
      "why are you so stupid ?!",
      "why you so serious ?!",
      "you are sooooo breauty !!!",
      "i am always doing the right things in bad time",
      "i am always doing the right things in bad time :(",
      "i love apple and hate nokia",
      "i want to sleep",
      "i am hungry",
      "i am so tried being here",
      "i am sad ,angry,hate every thing"]

 
 counter =0
 for i in yhat1 :
     if (i==1):
      print(X1[counter])
      counter = counter+1
      print("Positive")
     elif (i==0) :
      print(X1[counter])
      counter = counter+1
      print("Negative")
     else:
         print("Neutral")

 yhat2 = model_Lstm.predict(TrainingSentencesSequences)


 print(yhat2)
 '''



if __name__ == '__main__':
    main()
