import twitter as twt
import requests
import pandas as pd
import datetime as date
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

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
        wed = self.get_Wednesday()
        print(wed)
        Link = "https://www.quandl.com/api/v3/datasets/AAII/AAII_SENTIMENT.json?api_key=XsXNLg3263w9ksoCtkBB&start_date=" +wed
        re = requests.get(url=Link)
        obj = re.json()['dataset']
        m = pd.DataFrame(obj['data'], columns=obj['column_names'])*100
        para = str(m.loc[0, ['Bullish', 'Neutral', 'Bearish']]).split("\n")
        reply = para[0] + "% <br/>" + para[1]+ "% <br/>" + para[2] + "%"
        return reply
    def get_Wednesday(self):
        today_d = date.datetime.today()
        day = int(today_d.weekday())
        reduct = 0
        if day > 3:
            reduct = day - 3
        elif day < 3:
            reduct = day + 7 - 3
        wed_d = today_d - date.timedelta(days=reduct)
        return str(wed_d)

    def daily_stock_data(self):
        alpha_vantage_api_key = "AUFGD5JSWQD96T0M"
        from alpha_vantage.timeseries import TimeSeries
        # Your key here
        ts = TimeSeries(alpha_vantage_api_key)
        data, meta = ts.get_daily(symbol=str(self.ticker).split('#')[1])
        today = str(date.datetime.today() - date.timedelta(days= 1)).split(" ")[0]
        print(today)
        val = list(data[today].values())
        resp = "Open: " + str(val[0]) + "<br/>High: " + str(val[1]) + "<br/>Low: "
        resp = resp + str(val[2]) + "<br/>Close: " + str(val[3]) + "<br/>Volume: " + str(val[4])
        #Visualization
        data_pd = pd.DataFrame(data).T
        print(data_pd)
        convert_dict = {'1. open': float,
                        '2. high': float,
                        '3. low': float,
                        '4. close': float,
                        '5. volume': int,
                        }
        df = data_pd.astype(convert_dict)
        figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
        df['4. close'].plot()
        plt.tight_layout()
        plt.ylabel("Close Price")
        plt.xlabel("Date")
        plt.grid()
        plt.savefig('static/graph.jpeg',bbox_inches='tight')
        plt.show()
        resp = resp + '<br/><a href="/img" target="_blank"> >>CLICK HERE<< </a>'
        return resp
def main():
    st = Stock("#AAPL")
    print(st.investor_sentiment())
if __name__=="__main__":
    main()