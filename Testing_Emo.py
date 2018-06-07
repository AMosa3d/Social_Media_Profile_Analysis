from __future__ import print_function
import csv
import os
import sys
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM , Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from keras.models import Sequential
import re
from keras.utils import np_utils


def del_punctutation(s):
    return re.sub("[\.\t\,\:;\(\)\_\.!\@\?\&\--]", "", s, 0, 0)

def get_tokens_list(Data):

    stemmer = SnowballStemmer("english")
    tokensList = []
    stopWords = set(stopwords.words('english'))

    for i in range(1, len(Data)):
        #### to get the words from every sentences
        t = []
        tokens = nltk.word_tokenize(del_punctutation(Data[i].lower()))
        for token in tokens:
            if token not in stopWords:
                t.append(token)
        #### add all tokens of the tweets to list
        tokensList.append(t)

    return tokensList

def Train_Model(TrainingSentences, TrainingLabels):

    """HyperParameters"""
    maxWordsLengthPerSentence = 25
    wordVectorSize = 100

    #2 - create Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(TrainingSentences)

    vocab_size = len(tokenizer.word_index)+1

    #3 - convert corpus to sequences
    TrainingSentencesSequences = tokenizer.texts_to_sequences(TrainingSentences)

    #4 - pad sequences
    TrainingSentencesSequences = pad_sequences(TrainingSentencesSequences, maxlen=maxWordsLengthPerSentence)

    #5 - Load Glove model and build dict - completed and passed in the parameter
    embeddings_index = dict()
    f = open('glove.twitter.27B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        word = word.lower()
        try:
           coefs = asarray(values[1:], dtype='float32')
           embeddings_index[word] = coefs
        except:
           continue
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(TrainingLabels)
    encoded_Y = encoder.transform(TrainingLabels)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    LSTM_Model = Sequential()

    LSTM_Model.add(Embedding(vocab_size, wordVectorSize, weights=[embedding_matrix], input_length=maxWordsLengthPerSentence, trainable=False))
    LSTM_Model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    LSTM_Model.add(Dense(10, activation='sigmoid'))


    LSTM_Model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
###rmsprop  adam
    leng = round(len(TrainingSentences)*.8)
    LSTM_Model.fit(
        TrainingSentencesSequences[1:leng],
        dummy_y[1:leng],
                   epochs=20,
      #  validation_data=(TrainingSentencesSequences[700001:800000], TrainingLabels[700001:800000])
    )

    leng = leng+1
    loss, accuarcy = LSTM_Model.evaluate(
        TrainingSentencesSequences[leng:],
        dummy_y[leng:],
        batch_size=32
    )

    print('Test score : ', loss)
    print('Test accuracy : ', accuarcy)
    LSTM_Model.save("Emo.h5")




def LoadData():
    TrainingSentences = []
    TrainingLabels = []
    with open('Jan9-2012-tweets-clean2.csv', 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            TrainingSentences.append(row[0])
            TrainingLabels.append(row[1])


    return TrainingSentences, TrainingLabels

def main():


    TrainingSentences, TrainingLabels = LoadData()

    model = Train_Model(TrainingSentences, TrainingLabels)



if __name__ == '__main__':
    main()
