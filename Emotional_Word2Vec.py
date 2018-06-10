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
from collections import Counter
from numpy import asarray
from numpy import zeros

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
                   epochs=10,
                   validation_data=(TrainingSentencesSequences[leng:leng2], TrainingLabels[leng:leng2])
    )


    loss, accuarcy = LSTM_Model.evaluate(
        TrainingSentencesSequences[leng2:],
        TrainingLabels[leng2:],
        batch_size=32
    )

    print('Test score : ', loss)
    print('Test accuracy : ', accuarcy)

    S = [
        "I am happy",
        "I am tired",
        "I am sad",
        "I am afraid",
        "I am angry",

    ]

    S = tokenizer.texts_to_sequences(S)

    # 4 - pad sequences
    S = pad_sequences(S, maxlen=maxWordsLengthPerSentence)
    y = LSTM_Model.predict(S)
    y1 = LSTM_Model.predict_classes(S)
    y2 = LSTM_Model.predict_proba(S)

    print("XD")


    return LSTM_Model


def LoadData():
    TrainingSentences = []
    TrainingLabels = []
    '''
    with open('Jan9-2012-tweets-clean2.csv', 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if row[1] != '':
                TrainingSentences.append(row[2])
                TrainingLabels.append(row[1])
    '''

    '''with open('text_emotion.csv', 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if row[1] != '':
                TrainingSentences.append(row[3])
                TrainingLabels.append(row[1])
    '''
    with open('data.bak.csv', 'r', encoding="latin-1") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if row[1] != '':
                TrainingSentences.append(row[0])
                TrainingLabels.append(row[1])

    c = Counter(TrainingLabels)
    print(c)
    print("XD")

    perm = np.random.permutation(len(TrainingSentences))
    ShuffledSentences = []
    ShuffledLabels = []
    for i in range(len(perm)):
        index = perm[i]
        ShuffledSentences.append(TrainingSentences[index])
        ShuffledLabels.append(TrainingLabels[index])

    return ShuffledSentences, ShuffledLabels

def PreProcess(TrainingSentences, TrainingLabels):
    return

def main():

    """Loading model of WordEmbedding Technique"""
    WordEmbeddingModel = LoadWord2VecModel()

    TrainingSentences, TrainingLabels = LoadData()

    model = Train_Model(WordEmbeddingModel, TrainingSentences, TrainingLabels)




if __name__ == '__main__':
    main()

