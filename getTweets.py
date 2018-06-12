import tweepy
import csv


maxTweets=5

def get_tweets(userName):
    #authorize twitter using the keys that we got from dev.twitter
    consumer_key = "IibibQUcX7Ssy1bteWKXFN89G"
    consumer_secret = "XTu1aGLLJqmihDbHL9ofhM51GIfmSuezi64Jieakz6bmVh2hCc"
    access_token = "155015907-KCa96Y9ZXGxSqUTMPHdJ6kmi3pZoZhvjGG1i966A"
    access_token_secret = "mt1k6pWiEU5W7PVNib3YCDXFX7pAChiwCGPaLYhMbhOd4"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    all_tweets=[]
    #make request to get the max number of tweets(200 is the max)
    new_tweets = api.user_timeline(screen_name = userName,count=maxTweets)
    #save last 200 tweet
    all_tweets.extend(new_tweets)
    #save the id of the oldest tweet
    oldest = all_tweets[-1].id - 1


    #the return list of tweets shof anta el response w akhtar htrg3 eh w 7oto
    outtweets = [[tweet.text] for tweet in all_tweets]
    user_object = api.get_user(userName)
    avatar_url = user_object.profile_image_url
    avatar_url = avatar_url[0:len(avatar_url) - 11] + '.jpg'
    return avatar_url,outtweets


'''
    # convert the list to csv file
    with open('%s_tweets.csv' % userName, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text", "friends"])
        writer.writerows(outtweets)
    pass
'''
'''
if __name__ == '__main__':
    get_tweets("MahmoudHigazy")
'''