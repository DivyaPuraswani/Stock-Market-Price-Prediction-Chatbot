import numpy as np
import time
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import sys
import os

class ge_code:

    def __init__(self):
        self.params = {
            "batch_size": 20,  # 20<16<10, 25 was a bust
            "epochs": 5,
            "lr": 0.00010000,
            "time_steps": 60
        }
        self.PATH_TO_ML_DATA = "C:/Users/amari/PycharmProjects/StockBot"
        self.INPUT_PATH = self.PATH_TO_ML_DATA+"/inputs"
        self.OUTPUT_PATH = self.PATH_TO_ML_DATA+"/models"
        self.TIME_STEPS = self.params["time_steps"]
        self.BATCH_SIZE = self.params["batch_size"]


    def trim_dataset(self,mat,batch_size):
        """
        trims dataset to a size that's divisible by BATCH_SIZE
        """
        no_of_rows_drop = mat.shape[0]%batch_size
        if no_of_rows_drop > 0:
            return mat[no_of_rows_drop:]
        else:
            return mat


    def build_timeseries(self,mat, y_col_index):
        """
        Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
        number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
        :param mat: ndarray which holds the dataset
        :param y_col_index: index of column which acts as output
        :return: returns two ndarrays-- input and output in format suitable to feed
        to LSTM.
        """
        # total number of time-series samples would be len(mat) - TIME_STEPS
        dim_0 = mat.shape[0] - self.TIME_STEPS
        dim_1 = mat.shape[1]
        x = np.zeros((dim_0, self.TIME_STEPS, dim_1))
        y = np.zeros((dim_0,))
        print("dim_0",dim_0)
        for i in tqdm_notebook(range(dim_0)):
            x[i] = mat[i: self.TIME_STEPS+i]
            y[i] = mat[self.TIME_STEPS+i, y_col_index]
    #         if i < 10:
    #           print(i,"-->", x[i,-1,:], y[i])
        print("length of time-series i/o",x.shape,y.shape)
        return x, y



    def create_model(self, x_t):
        lstm_model = Sequential()
        # (batch_size, timesteps, data_dim)
        lstm_model.add(LSTM(100, batch_input_shape=(self.BATCH_SIZE, self.TIME_STEPS, x_t.shape[2]),
                            dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                            kernel_initializer='random_uniform'))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(LSTM(60, dropout=0.0))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(Dense(20,activation='relu'))
        lstm_model.add(Dense(1,activation='sigmoid'))
        optimizer = optimizers.RMSprop(lr=self.params["lr"])
        # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
        lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
        return lstm_model


    def execute(self):
        # stime = time.time()
        # print(os.listdir(INPUT_PATH))
        df_ge = pd.read_csv(os.path.join(self.INPUT_PATH, "GE.csv"), engine='python')
        # print(df_ge.shape)
        # print(df_ge.columns)
        # print(df_ge.head(5))
        # tqdm_notebook.pandas('Processing...')
        # df_ge = process_dataframe(df_ge)
        # print(df_ge.dtypes)
        train_cols = ["Open", "High", "Low", "Close", "Volume"]
        df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
        print("Train--Test size", len(df_train), len(df_test))

        # scale the feature MinMax, build array
        x = df_train.loc[:, train_cols].values
        min_max_scaler = MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x)
        x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

        print("Deleting unused dataframes of total size(KB)",
              (sys.getsizeof(df_ge) + sys.getsizeof(df_train) + sys.getsizeof(df_test)) // 1024)

        del df_ge
        del df_test
        del df_train
        del x

        print("Are any NaNs present in train/test matrices?", np.isnan(x_train).any(), np.isnan(x_train).any())
        x_t, y_t = self.build_timeseries(x_train, 3)
        x_t = self.trim_dataset(x_t, self.BATCH_SIZE)
        y_t = self.trim_dataset(y_t, self.BATCH_SIZE)
        print("Batch trimmed size", x_t.shape, y_t.shape)

        model = None
        try:
            model = pickle.load(open("models/lstm_model", 'rb'))
            print("Loaded saved model...")
        except FileNotFoundError:
            print("Model not found")


        x_temp, y_temp = self.build_timeseries(x_test, 3)
        x_val, x_test_t = np.split(self.trim_dataset(x_temp, self.BATCH_SIZE),2)
        y_val, y_test_t = np.split(self.trim_dataset(y_temp, self.BATCH_SIZE),2)

        print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)

        is_update_model = True
        if model is None:
            from keras import backend as K
            print("Building model...")
            print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
            model = self.create_model(x_t)

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=40, min_delta=0.0001)

            mcp = ModelCheckpoint(os.path.join(self.OUTPUT_PATH,
                                  "best_model.h5"), monitor='val_loss', verbose=1,
                                  save_best_only=True, save_weights_only=False, mode='min', period=1)

            # Not used here. But leaving it here as a reminder for future
            r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30,
                                          verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

            csv_logger = CSVLogger(os.path.join(self.OUTPUT_PATH, 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)

            history = model.fit(x_t, y_t, epochs=self.params["epochs"], verbose=2, batch_size=self.BATCH_SIZE,
                                shuffle=False, validation_data=(self.trim_dataset(x_val, self.BATCH_SIZE),
                                self.trim_dataset(y_val, self.BATCH_SIZE)), callbacks=[es, mcp, csv_logger])

            print("saving model...")
            pickle.dump(model, open("models/lstm_model", "wb"))
            # Visualize the training data
            from matplotlib import pyplot as plt
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            #plt.show()
            plt.savefig(os.path.join(self.OUTPUT_PATH, 'train_vis_BS_'+str(self.BATCH_SIZE)+"_"+time.ctime()+'.png'))

        # model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE
        y_pred = model.predict(self.trim_dataset(x_test_t, self.BATCH_SIZE), batch_size=self.BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = self.trim_dataset(y_test_t, self.BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)
        print(y_pred[0:15])
        print(y_test_t[0:15])

        # convert the predicted value to range of real data
        y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
        # min_max_scaler.inverse_transform(y_pred)
        y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
        # min_max_scaler.inverse_transform(y_test_t)
        print(y_pred_org[0:15])
        print(y_test_t_org[0:15])



        # load the saved best model from above
        saved_model = load_model(os.path.join(self.OUTPUT_PATH, 'best_model.h5')) # , "lstm_best_7-3-19_12AM",
        print(saved_model)

        y_pred = saved_model.predict(self.trim_dataset(x_test_t, self.BATCH_SIZE), batch_size=self.BATCH_SIZE)
        y_pred = y_pred.flatten()
        y_test_t = self.trim_dataset(y_test_t, self.BATCH_SIZE)
        error = mean_squared_error(y_test_t, y_pred)
        print("Error is", error, y_pred.shape, y_test_t.shape)
        print(y_pred[0:15])
        print(y_test_t[0:15])
        y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_pred)
        y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3] # min_max_scaler.inverse_transform(y_test_t)
        print(y_pred_org[-1])
        print(len(y_test_t_org[-10:]))
        print(y_test_t_org[-10:])
        y_test_15 = list(y_test_t_org[-15:])
        y_test_15.append(y_pred_org[-1])
        return str(y_pred_org[-1])

if __name__ == '__main__':
    g = ge_code()
    g.execute()