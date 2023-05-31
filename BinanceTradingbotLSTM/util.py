#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:05:24 2023

@author: be
"""
from CONFIG import api_key, api_secret
from tensorflow.keras import layers
import tensorflow as tf
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def getInfo():
    client = Client(api_key, api_secret)
    print(client.get_account())

        
def getBalance():
    data = {'coin':['BTC'], 'amount':[0.0], 'actVal':[0.0]}
    portfolio = pd.DataFrame(data)
    portfolio.set_index("coin", inplace = True)
    try:
        portfolio = pd.read_pickle("balance.txt")
    except Exception as error:
        portfolio.to_pickle("balance.txt")
        print(error)
    return portfolio
        
def getPrice(client, sym):
    currentTickers = client.get_all_tickers()
    
    def getFittingPrice(sym):
        for i in currentTickers:
            if sym == i["symbol"]:
                return float(i["price"])
    return getFittingPrice(sym)

def get_filter(symbol):
    client = Client(api_key, api_secret)
    data_from_api = client.get_exchange_info()
    symbol_info = next(filter(lambda x: x['symbol'] == symbol, data_from_api['symbols']))
    
    return symbol_info


def getModel():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=40, return_sequences=True, dropout=0.1))
    model.add(layers.LSTM(units=40, return_sequences=True, dropout=0.1))
    model.add(layers.LSTM(units=40, dropout=0.02))
    model.add(layers.Dense(units=1))
    return model

def trainModel(X_train, X_test, y_train, y_test):
    model = getModel()  
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, verbose = 1)
    
    return model


def buy(symbol, quantity):
   
    return sell(symbol = symbol, quantity=quantity, action = "BUY") 
   
    
def sell(symbol = "BTCUSDT", quantity = 0.0004, action = "SELL"):
    print(action, " ", symbol)
    client = Client(api_key, api_secret)
    p = getPrice(client, symbol)
    filters = get_filter(symbol)
    filters["filters"]
    minNotional = float(next(filter(lambda x: x['filterType'] == "NOTIONAL", filters["filters"]))["minNotional"])
    if minNotional > p * quantity:
        return False
    #q = "{:0.0{}f}".format(quantity, 4)
        
    buy_order_limit = client.create_test_order(symbol=symbol,
                                               side='BUY', type='LIMIT',
                                               timeInForce='GTC',
                                               quantity=quantity,
                                               price=p)
    return buy_order_limit
    
    

    
def plotData(data2, coin ="BNBBTC"):
    d = data2
    d.reset_index(drop=True, inplace=True)
    Y = d
    X = np.arange(0,Y.shape[0],1)
    plt.title(coin)
    plt.xlabel("time")
    plt.ylabel("close")
    plt.plot(X,Y)
    plt.show()
    
    
    
