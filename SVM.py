from sklearn import svm
import csv
import time
from sklearn.feature_extraction.text import TfidfVectorizer

TrainingSentences = []
TrainingLabels = []
TestingSentences = []
TestingLabels = []
t0 = time.time()

print("1) Reading Data ...")

with open('train.csv', 'r', encoding="latin-1") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        TrainingSentences.append(row[2])
        TrainingLabels.append(row[1])

with open('Sentiment Analysis Dataset.csv', 'r', encoding="latin-1") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        TestingSentences.append(row[3])
        TestingLabels.append(row[1])


print("2) Feature Extracting ...")
vectorizer = TfidfVectorizer()

TestingSentences = TestingSentences[1:5002]
TestingLabels = TestingLabels[1:5002]


TrainingSentences = vectorizer.fit_transform(TrainingSentences)


clf = svm.SVC(kernel='rbf')
print("3) Training and Fitting ...")
clf.fit(TrainingSentences, TrainingLabels)


print("4) Predicting ...")


t = vectorizer.transform(["I love you"])

correct = 0

for i in range(0, len(TestingSentences)):
    currentCorpse = [TestingSentences[i]]
    currentCorpse = vectorizer.transform(currentCorpse)
    predict = clf.predict(currentCorpse)
    if predict == TestingLabels[i]:
        correct = correct + 1

accuracy = correct/len(TestingLabels)

print(accuracy)

print(time.time() - t0)
