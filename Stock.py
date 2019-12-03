import twitter as twt
import requests
import pandas as pd
import datetime as date
import matplotlib.pyplot as plt
import pandas_datareader as pd_read
from lstm_model import lstm_model

class Stock:

    def __init__(self,ticker):
        self.ticker = ticker

    def get_stock_date(self, stock):
        start_date = '2018-01-01'
        end_date = str(date.datetime.today())
        print (stock)
        data = pd_read.data.DataReader(stock.split("#")[1], 'yahoo', start_date, end_date)
        data['Date'] = data.index
        return data

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
        Link = "https://www.quandl.com/api/v3/datasets/AAII/AAII_SENTIMENT.json?api_key=XsXNLg3263w9ksoCtkBB&start_date="
        re = requests.get(url=Link)
        obj = re.json()['dataset']
        m = pd.DataFrame(obj['data'], columns=obj['column_names']).head(1)*100
        para = str(m.loc[0, ['Bullish', 'Neutral', 'Bearish']]).split("\n")
        reply = para[0] + "% <br/>" + para[1]+ "% <br/>" + para[2] + "%"
        return reply
    # def get_Wednesday(self):
    #     today_d = date.datetime.today()
    #     day = int(today_d.weekday())
    #     reduct = 0
    #     if day > 3:
    #         reduct = day - 3
    #     elif day < 3:
    #         reduct = day + 7 - 3
    #     wed_d = today_d - date.timedelta(days=reduct)
    #     return str(wed_d)

    def daily_stock_data(self, stock2 = None):
        today = str(date.datetime.today())
        print(today)
        data = self.get_stock_date(self.ticker)
        val = data.values[-1:].tolist()
        resp = str("Open: " + str(val[0][0]) + "<br/>Low: " + str(val[0][1]) + "<br/>High: ")
        resp = resp + str(val[0][2]) + "<br/>Close: " + str(val[0][3]) + "<br/>Volume: " + str(val[0][4]) + "<br/>"
        #Visualization
        print(data)

        if stock2 != None:
            stock2 = "#"+stock2
            data2 = self.get_stock_date(stock2)
            data['Close'] = data['Close'] / data['Close'].max()
            data2['Close'] = data2['Close'] / data2['Close'].max()
            data2[self.ticker] = data['Close']
            data2[stock2] = data2['Close']
            data2.plot(y = [self.ticker,stock2], x = 'Date',grid = True, figsize=(15,6))
            title = "Standardized Close price:" + self.ticker + " vs " + stock2
            plt.title(title)
            resp = ""
        else:
            title = "Close price Graph:" + self.ticker
            plt.title(title)
            data.plot(y = 'Close', x = 'Date',grid = True, figsize=(15,6))
        random = str(date.datetime.today()).split(".")[1]
        src = "static/graph" + random + ".jpeg"
        plt.savefig(src)
        resp = resp + '<a href="/img" target="_blank"> >>CLICK HERE<< </a>' + random
        return resp
    def stock_predict(self):
        print(self.ticker)
        print("self.ticker:",self.ticker)
        l = lstm_model()
        return l.execute(self.ticker)


# def main():
#     st = Stock("#MSFT")
#     # print(st.daily_stock_data("GOOG"))
#     print("Predicted:",st.stock_predict())
#
# if __name__=="__main__":
#     main()