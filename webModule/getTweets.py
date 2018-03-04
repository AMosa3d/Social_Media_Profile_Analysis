import tweepy
import csv


consumer_key=""
consumer_secret=""
access_token=""
access_token_secret=""
maxTweets=200
def get_tweets(userName):
    #authorize twitter using the keys that we got from dev.twitter
    consumer_key="xxx"
    consumer_secret="xxx"
    access_token="xxx-xxx"
    access_token_secret="xxx"
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
    #keep loading tweets until there are no tweets left in this profile
    while len(new_tweets)>0:
        #get new tweets using the oldest id saved to prevent duplicates
        new_tweets = api.user_timeline(screen_name=userName, count=200,max_id=oldest)
        all_tweets.extend(new_tweets)
        oldest = all_tweets[-1].id - 1
        
    #the return list of tweets shof anta el response w akhtar htrg3 eh w 7oto
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in all_tweets]

    # convert the list to csv file
    with open('%s_tweets.csv' % userName, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text"])
        writer.writerows(outtweets)

    pass

if __name__ == '__main__':
    get_tweets("MahmoudHigazy")
