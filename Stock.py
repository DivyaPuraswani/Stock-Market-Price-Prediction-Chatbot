import twitter as twt
import requests
import pandas as pd
import datetime as date
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

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

    def daily_stock_data(self, stock2 = None):
        alpha_vantage_api_key = "AUFGD5JSWQD96T0M"
        # Your key here
        ts = TimeSeries(alpha_vantage_api_key)
        data, meta = ts.get_daily(symbol=str(self.ticker).split('#')[1])
        today = str(date.datetime.today() - date.timedelta(days= 1)).split(" ")[0]
        print(today)
        val = list(data[today].values())
        resp = "Open: " + str(val[0]) + "<br/>High: " + str(val[1]) + "<br/>Low: "
        resp = resp + str(val[2]) + "<br/>Close: " + str(val[3]) + "<br/>Volume: " + str(val[4] + "<br/>")
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

        if stock2 != None:
            data2, meta2 = ts.get_daily(symbol= stock2)
            data_pd2 = pd.DataFrame(data2).T
            df2 = data_pd2.astype(convert_dict)
            df['4. close'] = df['4. close'] / df['4. close'].max()
            df2['4. close'] = df2['4. close'] / df2['4. close'].max()
            df2['Date'] = df2.index
            df2.sort_values('Date', inplace=True)
            df2['4. close'].plot()
            title = "Standardized Close price:" + self.ticker + " vs " + stock2
            plt.title(title)
            resp = ""
        else:
            title = "Close price Graph:" + self.ticker
            plt.title(title)
        df['Date'] = df.index
        df.sort_values('Date', inplace=True)
        df['4. close'].plot()
        plt.tight_layout()
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.grid()
        random = str(date.datetime.today()).split(".")[1]
        src = "static/graph" + random + ".jpeg"
        plt.savefig(src, bbox_inches='tight')
        plt.show()
        resp = resp + '<a href="/img" target="_blank"> >>CLICK HERE<< </a>' + random
        return resp

def main():
    st = Stock("#AAPL")
    print(st.daily_stock_data("GOOG"))
if __name__=="__main__":
    main()