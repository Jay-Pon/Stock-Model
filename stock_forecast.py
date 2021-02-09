import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import requests
import os
import io

ALPHA_VANTAGE_API = os.environ.get("ALPHA_VANTAGE_API")

def createDataset(data, look_back = 10):
    X = []
    Y = []
    
    total = data.shape[0]
    data = list(data['adjusted_close']) 

    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i + look_back])
    
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    return X, np.array(Y)

def makeModel(X, Y):
    
    model = Sequential()
    model.add(LSTM(units = 10, activation='relu', input_shape=(1, X.shape[2])))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    model.fit(X, Y, epochs=75)
    
    return model

def getData(symb):
    alpha_call = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&datatype=csv'.format(symbol = symb, api_key = ALPHA_VANTAGE_API)
    content = requests.get(alpha_call).content
    data = pd.read_csv(io.StringIO(content.decode('utf-8')))
    data = data[['adjusted_close']]

    adj_price = data['adjusted_close'].max()
    min_price = data['adjusted_close'].min()

    data['adjusted_close'] = data['adjusted_close'].apply(lambda x : (x - data['adjusted_close'].min()) / (data['adjusted_close'].max() - data['adjusted_close'].min()))
    total = data.shape[0]
    num_train = int(total * 0.67)
    training = data[:num_train]
    test = data[num_train:]

    train_X, train_Y = createDataset(training, look_back = 10)
    test_X, test_Y = createDataset(test, look_back = 10)

    return min_price, adj_price, train_X, train_Y, test_X, test_Y

def getError(predictions, actual):
    error = 0
    for i in range(len(predictions)):
        error += abs(actual[i] - predictions[i])
    return error / len(predictions)

symbol = input("Enter a symbol: ")

min_price, adj_price, train_X, train_Y, test_X, test_Y = getData(symbol)
model = makeModel(train_X, train_Y)
predictions = model.predict(test_X)

print("Error: ", getError(predictions, test_Y) * adj_price)