import csv
from gensim.models import Word2Vec
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
import re
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os
from numpy import asarray
from numpy import zeros
import pickle
from keras.models import load_model
from keras.models import save_model

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
        print("Loading Word2Vec Model ...")
        model = Word2Vec.load('\Word2Vec_Model')
    except:
        wordsData = []
        print("Training Word2Vec Model ...")
        with open('unlabeledTrainData.tsv', 'r',
                  encoding="latin-1") as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            for row in readCSV:
                wordsData.append(row[1])

        wordsData = get_tokens_list(wordsData)

        model = Word2Vec(wordsData)
        model.save('\Word2Vec_Model')

    return model

def Train_Model(TrainingSentences, TrainingLabels, maxWordsLengthPerSentence):

    """HyperParameters"""

    wordVectorSize = 100


    #tokens = get_tokens_list(TrainingSentences)

    #2 - create Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(TrainingSentences)
    with open('emotional_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    vocab_size = len(tokenizer.word_index)+1

    #3 - convert corpus to sequences
    TrainingSentencesSequences = tokenizer.texts_to_sequences(TrainingSentences)

    #4 - pad sequences
    TrainingSentencesSequences = pad_sequences(TrainingSentencesSequences, maxlen=maxWordsLengthPerSentence)

    #5 - Load GloVe and build dict - completed and passed in the parameter
    embeddings_index = dict()
    f = open('glove.twitter.27B.100d.txt', encoding="utf8")
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



    encoder = LabelEncoder()
    encoder.fit(TrainingLabels)
    TrainingLabels = encoder.transform(TrainingLabels)
    TrainingLabels = np_utils.to_categorical(TrainingLabels)


    LSTM_Model = Sequential()

    LSTM_Model.add(Embedding(vocab_size, wordVectorSize, weights=[embedding_matrix], input_length=maxWordsLengthPerSentence, trainable=False))
    LSTM_Model.add(Bidirectional(LSTM(128, activation='relu', dropout=0.2, recurrent_dropout=0.2)))
    LSTM_Model.add(Dense(TrainingLabels.shape[1], activation='sigmoid'))


    LSTM_Model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    leng = round(len(TrainingSentences) * .6)
    leng2 = leng + round(len(TrainingSentences) * .2)
    LSTM_Model.fit(
        TrainingSentencesSequences[1:leng],
                   TrainingLabels[1:leng],
                   epochs=8,
                   validation_data=(TrainingSentencesSequences[leng:leng2], TrainingLabels[leng:leng2])
    )


    loss, accuarcy = LSTM_Model.evaluate(
        TrainingSentencesSequences[leng2:],
        TrainingLabels[leng2:],
        batch_size=32
    )

    print('Test score : ', loss)
    print('Test accuracy : ', accuarcy)

    LSTM_Model.save('emotional_model.h5')


    return LSTM_Model

def Test_Model(LSTM_Model, TestingSentences, tokenizer, maxWordsLengthPerSentence):



    TestingSentences = tokenizer.texts_to_sequences(TestingSentences)

    # 4 - pad sequences
    TestingSentences = pad_sequences(TestingSentences, maxlen=maxWordsLengthPerSentence)

    y1 = LSTM_Model.predict_classes(TestingSentences)

    return y1


def LoadData():
    TrainingSentences = []
    TrainingLabels = []

    with open('data.bak.csv', 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if row[1] != '':
                TrainingSentences.append(row[0])
                TrainingLabels.append(row[1])


    perm = np.random.permutation(len(TrainingSentences))
    ShuffledSentences = []
    ShuffledLabels = []
    for i in range(len(perm)):
        index = perm[i]
        ShuffledSentences.append(TrainingSentences[index])
        ShuffledLabels.append(TrainingLabels[index])

    return ShuffledSentences, ShuffledLabels

def LoadTrainedModel():
    with open('emotional_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = load_model('emotional_model.h5')

    return tokenizer, model

def main():
    maxWordsLengthPerSentence = 25


    if not os.path.exists('emotional_model.h5'):
        TrainingSentences, TrainingLabels = LoadData()
        model = Train_Model(TrainingSentences, TrainingLabels, maxWordsLengthPerSentence)

    tokenizer, model = LoadTrainedModel()

    TestingSentences = [
        "I am happy",
        "I am tired",
        "I am sad",
        "I am afraid",
        "I am angry",
    ]

    Labels = Test_Model(model, TestingSentences, tokenizer, maxWordsLengthPerSentence)
    Res = []

    for i in range(len(Labels)):
        if Labels[i] == 0:
            Res.append('Neutral')
        elif Labels[i] == 1:
            Res.append('Happy')
        elif Labels[i] == 2:
            Res.append('Sad')
        elif Labels[i] == 3:
            Res.append('Hate')
        elif Labels[i] == 4:
            Res.append('Anger')

    print(Res)


if __name__ == '__main__':
    main()

