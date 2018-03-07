import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


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
 tokens=nltk.word_tokenize(TrainingSentences[i].lower())
 for token in tokens:
     if (token not in stopWords):
         tt = stemmer.stem(token)
         t.append(tt)

 tokensList.append(t)




print(tokensList[4])

