import TESTLSTMMODEL
import Emotional_GloVe

def main():

    tweets = ["i hate mimo waked", "i love mimo"]

    yhat1 =TESTLSTMMODEL.main(tweets)
    counter = 0
    for i in yhat1:
        if (i == 1):
            print(tweets[counter])
            counter = counter + 1
            print("Positive")
        elif (i == 0):
            print(tweets[counter])
            counter = counter + 1
            print("Negative")
        else:
            print("Neutral")

    resEmo = Emotional_GloVe.main(tweets)
    print(resEmo)

if __name__ == '__main__':
    main()

