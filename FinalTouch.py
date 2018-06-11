import TESTLSTMMODEL
import Emotional_GloVe
import getTweets
import re

def del_Punctutation(s):
    return re.sub(r'^https?://.[\r\n]', '',s, flags=re.MULTILINE)

def main():

    tweets = getTweets.get_tweets('@Ah_Samir1907')
    tweets = [del_Punctutation(tweet[0]) for tweet in tweets]

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

