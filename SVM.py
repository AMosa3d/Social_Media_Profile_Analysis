from sklearn import svm
import csv
import time
from sklearn.feature_extraction.text import TfidfVectorizer

X = []
Y = []

t0 = time.time()

print("1) Reading Data ...")
with open('Sentiment Analysis Dataset.csv', 'r', encoding="utf8") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    for row in readCSV:
        X.append(row[3])
        Y.append(row[1])

print("2) Feature Extracting ...")
vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.8, sublinear_tf=True, use_idf=True)

print("3) Training ...")
X = vectorizer.fit_transform(X)
#Y = vectorizer.fit_transform(Y)

clf = svm.SVC()
print("4) Fitting ...")
clf.fit(X[2:100000], Y[2:100000])


print("5) Predicting ...")
t = vectorizer.transform(["Have played by Howard Elliot thingy album today.  I really like it.  Well done me"])
hobba = clf.predict(t)
print(hobba)
print(time.time() - t0)
