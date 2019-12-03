import pandas_datareader as pd_read
import time  # helper libraries
import numpy as np
import math
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import datetime

class lstm_model:

    def get_normalised_data(self, data):
        """
        Normalises the data values using MinMaxScaler from sklearn
        :param data: a DataFrame with columns as  ['index','Open','Close','Volume']
        :return: a DataFrame with normalised value for all the columns except index
        """
        # Initialize a scaler, then apply it to the features
        scaler = MinMaxScaler()
        numerical = ['Open', 'Close', 'Volume']
        data[numerical] = scaler.fit_transform(data[numerical])

        return data


    def remove_data(self, data):
        """
        Remove columns from the data
        :param data: a record of all the stock prices with columns as  ['Date','Open','High','Low','Close','Volume']
        :return: a DataFrame with columns as  ['index','Open','Close','Volume']
        """
        # Define columns of data to keep from historical stock data
        item = []
        open = []
        close = []
        volume = []

        # Loop through the stock data objects backwards and store factors we want to keep
        i_counter = 0
        for i in range(len(data) - 1, -1, -1):
            item.append(i_counter)
            open.append(data['Open'][i])
            close.append(data['Close'][i])
            volume.append(data['Volume'][i])
            i_counter += 1

        # Create a data frame for stock data
        stocks = pd.DataFrame()

        # Add factors to data frame
        stocks['Item'] = item
        stocks['Open'] = open
        stocks['Close'] = pd.to_numeric(close)
        stocks['Volume'] = pd.to_numeric(volume)

        # return new formatted data
        return stocks



    def scale_range(self, x, input_range, target_range):
        """
        Rescale a numpy array from input to target range
        :param x: data to scale
        :param input_range: optional input range for data: default 0.0:1.0
        :param target_range: optional target range for data: default 0.0:1.0
        :return: rescaled array, incoming range [min,max]
        """
        range = [np.amin(x), np.amax(x)]
        x_std = (x - input_range[0]) / (1.0 *(input_range[1] - input_range[0]))
        x_scaled = x_std * (1.0 *(target_range[1] - target_range[0])) + target_range[0]
        return x_scaled, range


    def train_test_split_lstm(self, stocks, prediction_time=1, test_data_size=450, unroll_length=50):
        """
            Split the data set into training and testing feature for Long Short Term Memory Model
            :param stocks: whole data set containing ['Open','Close','Volume'] features
            :param prediction_time: no of days
            :param test_data_size: size of test data to be used
            :param unroll_length: how long a window should be used for train test split
            :return: X_train : training sets of feature
            :return: X_test : test sets of feature
            :return: y_train: training sets of label
            :return: y_test: test sets of label
        """
        # training data
        test_data_cut = test_data_size + unroll_length + 1

        x_train = stocks[0:-prediction_time - test_data_cut].values
        y_train = stocks[prediction_time:-test_data_cut]['Close'].values

        # test data
        x_test = stocks[0 - test_data_cut:-prediction_time].values
        y_test = stocks[prediction_time - test_data_cut:]['Close'].values

        return x_train, x_test, y_train, y_test


    def unroll(self, data, sequence_length=24):
        """
        use different windows for testing and training to stop from leak of information in the data
        :param data: data set to be used for unrolling
        :param sequence_length: window length
        :return: data sets with different window.
        """
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)


    def build_improved_model(self, input_dim, output_dim, return_sequences):
        """
        Builds an improved Long Short term memory model using keras.layers.recurrent.lstm
        :param input_dim: input dimension of model
        :param output_dim: ouput dimension of model
        :param return_sequences: return sequence for the model
        :return: a 3 layered LSTM model
        """
        model = Sequential()
        model.add(LSTM(
            input_shape=(None, input_dim),
            units=output_dim,
            return_sequences=return_sequences))

        model.add(Dropout(0.2))

        model.add(LSTM(
            128,
            return_sequences=False))

        model.add(Dropout(0.2))

        model.add(Dense(
            units=1))
        model.add(Activation('linear'))

        return model


    def price(self,x):
        """
        format the coords message box
        :param x: data to be formatted
        :return: formatted data
        """
        return '$%1.2f' % x


    def plot_prediction(self, actual, prediction, title='Google Trading vs Prediction', y_label='Price USD', x_label='Trading Days'):
        """
        Plots train, test and prediction
        :param actual: DataFrame containing actual data
        :param prediction: DataFrame containing predicted values
        :param title:  Title of the plot
        :param y_label: yLabel of the plot
        :param x_label: xLabel of the plot
        :return: prints a Pyplot againts items and their closing value
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Add labels
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        # Plot actual and predicted close values

        plt.plot(actual, '#00FF00', label='Adjusted Close')
        plt.plot(prediction, '#0000FF', label='Predicted Close')

        # Set title
        ax.set_title(title)
        ax.legend(loc='upper left')

        plt.show()


    def plot_lstm_prediction(self, actual, prediction, title='Google Trading vs Prediction', y_label='Price USD', x_label='Trading Days'):
        """
        Plots train, test and prediction
        :param actual: DataFrame containing actual data
        :param prediction: DataFrame containing predicted values
        :param title:  Title of the plot
        :param y_label: yLabel of the plot
        :param x_label: xLabel of the plot
        :return: prints a Pyplot againts items and their closing value
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Add labels
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        # Plot actual and predicted close values

        plt.plot(actual, '#00FF00', label='Adjusted Close')
        plt.plot(prediction, '#0000FF', label='Predicted Close')

        # Set title
        ax.set_title(title)
        ax.legend(loc='upper left')



        # plt.show()
        plt.savefig('actual vs pred.png')

    #Calling fucntions

    def execute(self, ticker):
        ticker = ticker.split("#")[1]
        start_date = '2000-01-01'
        end_date = str(datetime.datetime.today())
        data = pd_read.data.DataReader(ticker,'yahoo',start_date,end_date)

        #data
        data['Date'] = data.index
        data = data.sort_index(ascending = False)
        #
        # data.to_csv('inputs/MSFT.csv',index = False)
        # data = pd.read_csv('inputs/MSFT.csv')

        ####
        stocks = self.remove_data(data)
        x_scaler = MinMaxScaler()
        numerical = ['Open', 'Close', 'Volume']
        stocks[numerical] = x_scaler.fit_transform(stocks[numerical])

        ####
        # stocks = pd.read_csv('inputs/MSFT_preprocessed.csv')
        stocks_data = stocks.drop(['Item'], axis =1)
        print(stocks_data.head())

        #Train Test Split

        X_train, X_test,y_train, y_test = self.train_test_split_lstm(stocks_data, 5)

        unroll_length = 50
        X_train = self.unroll(X_train, unroll_length)
        X_test = self.unroll(X_test, unroll_length)
        y_train = y_train[-X_train.shape[0]:]
        y_test = y_test[-X_test.shape[0]:]

        print("x_train", X_train.shape)
        print("y_train", y_train.shape)
        print("x_test", X_test.shape)
        print("y_test", y_test.shape)

        model = None
        src = "models/" + ticker
        batch_size = 100
        try:
            model = pickle.load(open(src, 'rb'))
            print("Loaded saved model...")
        except:
            print("Model not found")
            # Set up hyperparameters
            epochs = 5

            is_update_model = True
            if model is None:
                model = self.build_improved_model( X_train.shape[-1] ,output_dim = unroll_length, return_sequences=True)

                start = time.time()
                # final_model.compile(loss='mean_squared_error', optimizer='adam')
                model.compile(loss='mean_squared_error', optimizer='adam')
                print('compilation time : ', time.time() - start)

                model.fit(X_train,
                          y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=2,
                          validation_split=0.05
                          )
                print("saving model...")
                pickle.dump(model, open(src, "wb"))

        # Generate predictions
        predictions = model.predict(X_test, batch_size=batch_size)
        #self.plot_lstm_prediction(y_test,predictions)
        #Evaluation
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

        #denormalised prediction
        pre = pd.DataFrame([],columns= ['Open', 'Close','Volume'])
        pre['Close'] = list(predictions)
        pre[numerical] = x_scaler.inverse_transform(pre[numerical])
        ndpredict = pre['Close'].tail(1).values[0]
        print(ndpredict)

        #range = [np.amin(stocks_data['Close']), np.amax(stocks_data['Close'])]

        # Calculate the stock price delta in $

        #true_delta = testScore *(range[1 ] -range[0])
        #print('Delta Price: %.6f - RMSE * Adjusted Close Range' % true_delta)

        #print(predictions[:15])
        data = data.sort_index(ascending=True)
        date = list(data['Date'].tail(15).values)
        print(date)
        add = 1
        m = [d.date() for d in pd.to_datetime(date)]
        x = m[14]
        if x.weekday() == 4:
            add = 3

        n_d = x + datetime.timedelta(days=add)
        m.append(n_d)
        y_test_15 = list(data['Close'].tail(15).values)
        y_test_15.append(ndpredict)
        direc = {'Date': m, 'Close': y_test_15}
        df = pd.DataFrame(direc, columns=['Date', 'Close'])
        print(df)
        df.plot(y='Close', x='Date', grid=True, figsize=(15, 6))
        title = "Close price Graph:" + ticker
        plt.title(title)
        plt.annotate("Predicted Close Price", xy=(m[15], y_test_15[15]), xytext=(m[13], y_test_15[15]),
                     arrowprops={'facecolor': 'black'})
        plt.legend(['Close Price'], loc='upper left')
        random = str(datetime.datetime.today()).split(".")[1]
        src = "static/graph" + random + ".jpeg"
        plt.savefig(src)
        plt.show()
        resp =  str(ndpredict)+'<br/><a href="/img" target="_blank"> >>CLICK HERE<< </a>' + random
        return resp

if __name__ == '__main__':
    g = lstm_model()
    print(g.execute("#MSFT"))