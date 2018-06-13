import io
import nltk
import numpy as np
from validation import compute_f1
from keras.models import Model,load_model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from prepro import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding
from keras.utils import plot_model,Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
from prepro import padding,createBatches

def main(tweets):

    model=load_model("NER.h5")
    tweetsList=[]
    ne_chunked_sents_list=[]
    for tweet in tweets:
        tokenized_doc = nltk.word_tokenize(tweet)
        tagged_sentences = nltk.pos_tag(tokenized_doc)
        ne_chunked_sents = nltk.ne_chunk(tagged_sentences)
        tweetsList.append(tokenized_doc)
        ne_chunked_sents_list.append(ne_chunked_sents)
    word2Idx = {}
    f = io.open("embeddings/glove.6B.100d.txt", encoding="utf-8")
    for line in f:
        split = line.strip().split(" ")
        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        word2Idx[split[0].lower()] = len(word2Idx)

    char2Idx = {"PADDING": 0, "UNKNOWN": 1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)

    word_input=[[]]
    words=tweet.split()
    charInd=[]
    wordInd=[]
    res=[]
    named_entities = []
    for word in words:
        if word.lower() in word2Idx:
            wordIdx=word2Idx[word.lower()]
        else:
            wordIdx=word2Idx["UNKNOWN_TOKEN"]
        temp_char=[]
        for char in word:
            temp_char.append(char2Idx[char])
        charInd.append(temp_char)
        wordInd.append(wordIdx)
    res.append([wordInd,charInd])
    res=padding(res)
    i=0;
    for ne_chunked_sents in ne_chunked_sents_list:
        named_entities.append([])
        for element in ne_chunked_sents:
            if hasattr(element, 'label'):
                entity_name = ' '.join(c[0] for c in element.leaves())
                #entity_type = element.label()  # get NE category
                named_entities[i].append((entity_name))
        i+=1
    ans=[]
    for i in res:
        for j in range(len(i[0])) :
            tokens=np.asarray([i[0][j]])
            char=np.asarray([i[1][j]])
            pred = model.predict([[tokens],[char]], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            if pred==0 or pred==1:
                ans.append("ORG")
            elif pred==3 or pred==8:
                ans.append("LOC")
            elif pred==4 or pred==7:
                ans.append("PER")
            else:
                ans.append("O")

    return named_entities


if __name__ == '__main__':
    x="Uruguay national team will play vs Egypt in Central Stadium we hope mo Salah Score A Goal"
    y="i love Alexandra"
    z="one of the greatest doctors in Ainshams universty is Abdelbadea"

    tweet=[]
    tweet.append(x)
    tweet.append(y)
    tweet.append(z)
    print main(tweet)