from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import csv
from keras.preprocessing.sequence import pad_sequences

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

def main():
 maxWordsLengthPerSentence = 25
 TrainingSentences, TrainingLabels = LoadData()

 # load model from single file

 model_Lstm = load_model('Shady3.h5')
 tokenizer = Tokenizer()
 tokenizer.fit_on_texts(TrainingSentences)

 # make predictions
 X1 =["I Love Eating Apple","I Hate Eating Apple",
      "i am so sad for my ex","i love sitting with myself",
      "you don't have a heart",
      "What is your worst Pokemon?",
      "What is your worst memory?",
      "work is sweet pain",
      "i am human",
      "i am soooo happpy",
      "i am soooo saaaad",
      "i am soooo saaaadddd",
      "What is your name?"]

 TrainingSentencesSequences = tokenizer.texts_to_sequences(X1)
 TrainingSentencesSequences = pad_sequences(TrainingSentencesSequences, maxlen=25)
 yhat1 = model_Lstm.predict_classes(TrainingSentencesSequences)
 yhat2 = model_Lstm.predict(TrainingSentencesSequences)
 for i in yhat1 :
     if (i==1):
      print("Positive")
     else :
      print("Negative")

 print(yhat2)




if __name__ == '__main__':
    main()
