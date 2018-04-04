import csv
from scipy import sparse
from gensim import corpora
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import numpy as np
import itertools

def del_Punctutation(s):
    return re.sub("[\.\t\,\:;\(\)\_\.!\@\?\&\--]", "", s, 0, 0)

def main():
 ##### Read The DataSet & apeend it in 2 list
 TrainingSentences=[]
 TrainingLabels=[]
 with open('Sentiment Analysis Dataset 100000.csv', 'r', encoding="latin-1") as csvfile:
     readCSV = csv.reader(csvfile, delimiter=",")
     for row in readCSV:
         TrainingSentences.append(row[2])
         TrainingLabels.append(row[1])

 ####### Pre Processing to clean The sentences
 print(TrainingSentences[5])
 stemmer = SnowballStemmer("english")
 tokensList=[]
 stopWords = set(stopwords.words('english'))

 for i in range(1,len(TrainingSentences)):
 #### to get the words from every sentences
  t=[]
  tokens=nltk.word_tokenize(del_Punctutation(TrainingSentences[i].lower()))
  for token in tokens:
      if (token not in stopWords):
          tt = stemmer.stem(token)
          t.append(tt)
#### add all tokens of the tweets to list
  tokensList.append(t)

# get frequence of wards in tokensList
 vocab = Counter()
 for line in tokensList:
     vocab.update(line)

#convert list vocab to dict
 Dic_vocab={word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}
#put a ID to every ward in the token List
 id2word = dict((i, word) for word, (i, _) in Dic_vocab.items())

 window_size = 10
 vocab_size = len(Dic_vocab)
 #########
 cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                   dtype=np.float64)

 for i, line in enumerate(Dic_vocab):
     token_ids = [Dic_vocab[word][0]for word in tokens]
     #get words to the left of the word
     for center_i, center_id in enumerate(token_ids):
     # Collect all word IDs in left window of center word
      context_ids = token_ids[
                    max(0, center_i - window_size)
                                    : center_i
                    ]
      contexts_len = len(context_ids)

     for left_index, left_id in enumerate(context_ids):

         #adding a weight to each word in the co-occurrence while a bigger weight is to the nearest word
         increment_weight = 1/float(contexts_len - left_index)

         # update the co-occurrence matrix in the both dimensions to make it also similar
         cooccurrences[center_id, left_id] += increment_weight
         cooccurrences[left_id, center_id] += increment_weight




#def train_glove(vocab, cooccurrences):
#vocab_size = len(vocab)
 vector_size = 100
 iterations = 1000
 learning_rate = 0.2

 W = ((np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1))

 Biases = ((np.random.rand(vocab_size*2) - 0.5) / float(vector_size+1))

 gradient_square = np.ones((vocab_size*2,vector_size),dtype=np.float64)

 gradient_square_biases = np.ones((vocab_size * 2), dtype=np.float64)

 data = []

 for i in range(iterations):
     cost = run_iteration(vocab,data,learning_rate)


def run_iteration(vocab,data,**kwargs):


##################################################
 print(tokensList[4])

if __name__ == '__main__':
 main()

