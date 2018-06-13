import TESTLSTMMODEL
import Emotional_GloVe
import getTweets
import re
import Report_Generator

def del_Punctutation(s):
    return re.sub(r'^https?://.[\r\n]', '',s, flags=re.MULTILINE)

def main(handle):
    Keywords = ["apple", "samsung", "Emam", "shady","twitch","witcher","spotify","Facebook","Youtube","Google","Pycharm",
                "Canda","London","Egypt","Pairs"]
    avatar_url,tweets = getTweets.get_tweets(handle)
    tweets = [del_Punctutation(tweet[0]) for tweet in tweets]
    yhat1 =TESTLSTMMODEL.main(tweets)
    Pos_Neg_Res = []
    for i in yhat1:
        if (i == 1):
            Pos_Neg_Res.append("Positive")
        elif (i == 0):
            Pos_Neg_Res.append("Negative")

    resEmo = Emotional_GloVe.main(tweets)
    Result_url=Report_Generator.main(tweets, resEmo, Pos_Neg_Res, Keywords, handle, avatar_url)
    return(Result_url)

