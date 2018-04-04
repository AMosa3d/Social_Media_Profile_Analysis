import csv
from gensim.models import Word2Vec
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import re

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

def LoadWord2VecModel():

    try:
        print("Training Word2Vec Model ...")
        model = Word2Vec.load('\Word2Vec_Model')
    except:
        wordsData = []
        print("Loading Word2Vec Model ...")
        with open('unlabeledTrainData.tsv', 'r',
                  encoding="latin-1") as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            for row in readCSV:
                wordsData.append(row[1])

        wordsData = get_tokens_list(wordsData)

        model = Word2Vec(wordsData)
        model.save('\Word2Vec_Model')

    return model

def Train_Model(WordEmbeddingModel, TrainingSentences, TrainingLabels):

    """HyperParameters"""
    maxWordsLengthPerSentence = 25
    wordVectorSize = 100


    #tokens = get_tokens_list(TrainingSentences)

    #2 - create Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(TrainingSentences)

    vocab_size = len(tokenizer.word_index)+1

    #3 - convert corpus to sequences
    TrainingSentencesSequences = tokenizer.texts_to_sequences(TrainingSentences)

    #4 - pad sequences
    TrainingSentencesSequences = pad_sequences(TrainingSentencesSequences, maxlen=maxWordsLengthPerSentence)

    #5 - Load Word2Vec and build dict - completed and passed in the parameter
    WordsList = []
    for line in WordEmbeddingModel.wv.index2word:
        WordsList.append(line)

    WordsVector = []
    for vector in WordEmbeddingModel.wv.vectors:
        WordsVector.append(vector)

    dictionary = dict()
    for i in range(len(WordsVector)):
        dictionary[WordsList[i]] = WordsVector[i]

    embedding_matrix = np.zeros((vocab_size, wordVectorSize))

    for word, i in tokenizer.word_index.items():
        vector = dictionary.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

    LSTM_Model = Sequential()

    LSTM_Model.add(Embedding(vocab_size, wordVectorSize, weights=[embedding_matrix], input_length=maxWordsLengthPerSentence, trainable=False))
    LSTM_Model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    LSTM_Model.add(Dense(1, activation='sigmoid'))


    LSTM_Model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    LSTM_Model.fit(
        TrainingSentencesSequences[1:80000],
                   TrainingLabels[1:80000],
                   epochs=15
                   #validation_data=(TrainingSentencesSequences[90001:], TrainingLabels[90001:])
    )


    loss, accuarcy = LSTM_Model.evaluate(
        TrainingSentencesSequences[90001:],
        TrainingLabels[90001:],
        batch_size=32
    )

    print('Test score : ', loss)
    print('Test accuracy : ', accuarcy)


def LoadData():
    TrainingSentences = []
    TrainingLabels = []
    with open('Sentiment Analysis Dataset 100000.csv', 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            TrainingSentences.append(row[2])
            TrainingLabels.append(row[1])


    TrainingSentences = TrainingSentences[1:]
    TrainingLabels = [int(x) for x in TrainingLabels[1:]]
    return TrainingSentences, TrainingLabels

def PreProcess(TrainingSentences, TrainingLabels):
    return

def main():

    """Loading model of WordEmbedding Technique"""
    WordEmbeddingModel = LoadWord2VecModel()

    TrainingSentences, TrainingLabels = LoadData()

    model = Train_Model(WordEmbeddingModel, TrainingSentences, TrainingLabels)


if __name__ == '__main__':
    main()

