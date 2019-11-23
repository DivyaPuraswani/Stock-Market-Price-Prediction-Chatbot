import twitter as twt
import requests
import pandas as pd

class Stock:

    def __init__(self,ticker):
        self.ticker = ticker

    def tweet_Sentiment(self):
        # creating object of TwitterClient Class
        api = twt.TwitterClient()
        # calling function to get tweets
        tweets = api.get_tweets(query=self.ticker, count=200)

        # picking positive tweets from tweets
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        # percentage of positive tweets
        pt = str("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
        print(pt)
        # picking negative tweets from tweets
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        # percentage of negative tweets
        nt = str("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
        print(nt)
        # percentage of neutral tweets
        nut = str("Neutral tweets percentage: {} % ".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))
        print(nut)

        # # printing first 5 positive tweets
        # print("\n\nPositive tweets:")
        # for tweet in ptweets[:10]:
        #     print(tweet['text'])
        #
        #     # printing first 5 negative tweets
        # print("\n\nNegative tweets:")
        # for tweet in ntweets[:10]:
        #     print(tweet['text'])
        return str(pt + '<br />' + nt + '<br />' + nut)
    def investor_sentiment(self):
        re = requests.get(url="https://www.quandl.com/api/v3/datasets/AAII/AAII_SENTIMENT.json?api_key=XsXNLg3263w9ksoCtkBB&start_date=2019-11-21")
        obj = re.json()['dataset']
        m = pd.DataFrame(obj['data'], columns=obj['column_names'])*100
        para = str(m.loc[0, ['Bullish', 'Neutral', 'Bearish']]).split("\n")
        reply = para[0] + "% <br/>" + para[1]+ "% <br/>" + para[2] + "%"
        return reply