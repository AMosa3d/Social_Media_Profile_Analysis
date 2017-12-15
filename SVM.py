from sklearn import svm

X = [[0,0], [1,1]]

Y = [2,3]

clf = svm.SVC()

clf.fit(X,Y)

hobba = clf.predict([[0.9,0.9]])
print(hobba)