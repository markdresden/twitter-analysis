#Source: https://www.youtube.com/watch?v=jtIMnmbnOFo
import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

query = "(from:FollowStevens)" #Change the query based on Twitter search criteria
tweets = []
limit = 5

def analysis(tweet):
    tweet2 = tweet.content
    print(tweet2)
    # precprcess tweet
    tweet_words = []

    for word in tweet2.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    for i in range(len(scores)):
        
        l = labels[i]
        s = scores[i]
        print(l,s)
    return scores


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        scores = analysis(tweet)
        tweets.append([tweet.date, tweet.user.username, tweet.content, scores])
        



df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'Scores'])
print(df)
df.to_csv('tweets.csv')


