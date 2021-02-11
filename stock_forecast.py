import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import requests
import os
import io

from secrets import ALPHA_VANTAGE_API

def createDataset(data, look_back = 10):
    X = []
    Y = []
    data = list(data['adjusted_close']) 

    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        Y.append(data[i + look_back])
    
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    return X, np.array(Y)

def makeModel(X, Y):
    model = Sequential()
    
    model.add(LSTM(units = 10, activation='relu', return_sequences = True, input_shape=(1, X.shape[2])))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 40, return_sequences = True, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 60, activation='relu'))
    model.add(Dropout(0.2))
        
    model.add(Dense(1))
    
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    model.fit(X, Y, epochs=75, verbose = 0)
    
    return model

def getData(symb, look_back = 10):
    alpha_api_call = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&datatype=csv'.format(symbol = symb, api_key = ALPHA_VANTAGE_API)
    content = requests.get(alpha_api_call).content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
    data = data.iloc[::-1]
    data = data[['adjusted_close']]
    adj_price = data['adjusted_close'].max()
    min_price = data['adjusted_close'].min()

    data['adjusted_close'] = data['adjusted_close'].apply(lambda x : (x - data['adjusted_close'].min()) / (data['adjusted_close'].max() - data['adjusted_close'].min()))
    total = data.shape[0]
    num_train = int(total * 0.67)
    training = data[:num_train]
    test = data[num_train:]

    train_X, train_Y = createDataset(training, look_back = look_back)
    test_X, test_Y = createDataset(test, look_back = look_back)

    return min_price, adj_price, train_X, train_Y, test_X, test_Y

def drive(symbol):
    min_price, adj_price, train_X, train_Y, test_X, test_Y = getData(symbol, look_back = 20)

    model = makeModel(train_X, train_Y)

    predictions = model.predict(test_X)

    print("Model error: ", mse(predictions, test_Y))

    tomorrow_data = test_X[-1:]
    tomorrow_price = model.predict(tomorrow_data) * (adj_price - min_price) + min_price
 
    print("Model 1 Prediction: ", tomorrow_price[0][0])

symbol = input("Enter a symbol: ")
drive(symbol)