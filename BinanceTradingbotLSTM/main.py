#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:42:58 2023

@author: be
"""
from binance.client import Client
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time


import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import random
from CONFIG import *
from loadStoreModels import *
from util import *
length = 120

#%%Get Balance of Account

portfolio = getBalance()
#%% Connection with Binance

client = Client(api_key, api_secret)

def getAllCoins():
    data = client.get_exchange_info()
    results = []
    for item in data["symbols"]:
        s = item['symbol']
        tr = item['status']
        if(tr == 'TRADING' and s.find("USDT") != -1 and s.find("USDC") == -1  and s.find("TUSD") == -1 
           and s.find("BUSD") == -1
           and s.find("USDP") == -1
           and s.find("USDS") == -1):
            
            results.append(item['symbol'])
        
    return results

def GetHistoricalData(howLong = 365 * 8, coin = "BNBBTC"):
    # Calculate the timestamps for the binance api function
    untilThisDate = datetime.datetime.now()
    sinceThisDate = untilThisDate - datetime.timedelta(days = howLong)
    # Execute the query from binance - timestamps must be converted to strings !
    candle = client.get_historical_klines(coin, Client.KLINE_INTERVAL_4HOUR, "1 Jan, 2017")
    #klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")


    # Create a dataframe to label all the columns returned by binance so we work with them later.
    df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    # as timestamp is returned in ms, let us convert this back to proper timestamps.
    #df.dateTime = pd.to_datetime(df.dateTime, unit='ms').dt.strftime(Constants.DateTimeFormat)
    df.set_index('dateTime', inplace=True)

    # Get rid of columns we do not need
    df = df.drop(['closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
    print("DataStart")
    print (df.iloc[-1])
    #df.apply(pd.to_numeric)
    return df.apply(pd.to_numeric)


def getData(coin ="BNBBTC"):
    return GetHistoricalData(coin = coin).iloc[:, 0]

def getPreprocessedData(d):
    data = d
    sc = MinMaxScaler()
    
    data.reset_index(drop=True, inplace=True)
    hist = []
    target = []
    for i in range(len(data)-length):
        x = data[i:i+length]
        y = data[i+length]
        hist.append(x)
        target.append(y)
    last = data[-length:]
    hist = np.array(hist)
    target = np.array(target)
    target = target.reshape(-1,1)
    if hist.shape[0] > 10:
       
        hist_scaled = sc.fit_transform(hist)
        last = sc.transform(np.array(last).reshape(1,length))
        target_scaled = sc.fit_transform(target)
       
        hist_scaled = hist_scaled.reshape((len(hist_scaled), length, 1))
        
        #split = int(hist_scaled.shape[0]*0.9)
        # Wähle alles aus
        split = hist_scaled.shape[0]
        
        X_train = hist_scaled[: split,:,:]
        X_test = hist_scaled[ split:,:,:]
        y_train = target_scaled[: split,:]
        y_test = target_scaled[ split:,:]
        X_train, X_test, y_train, y_test, sc
        return X_train, X_test, y_train, y_test, last, sc
    return  np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), sc

#%% Start Training the models
predDict = dict()
modelDict = dict()

for i in getAllCoins():
   coin = str(i)
   
   print("Currently working with ", coin)
   #plotData(coin)
   e = True
   
   while(e):
       try:
           data2 = getData(coin)
           X_train = []
           X_train, X_test, y_train, y_test, last,  sc = getPreprocessedData(data2)
           e = False
       except Exception as error:
           print(error)
           print("NextTry")
           time.sleep(4 + int(random.random() * 5))
   try:
        X_train
        wellDefined = True
   except NameError:
        wellDefined = False

   if wellDefined and X_train.shape[0] > 400:
       plotData(data2, coin = coin)
       model = trainModel(X_train, X_test, y_train, y_test)
       modelDict[coin] = model
       saveModels(coin, model, [sc.data_min_[0],sc.data_max_[0]])
       #print("Evaluation:")
       #model.evaluate(X_test,y_test)
       
       # Predict with newest Data

       pred = model.predict(last, verbose = 0)
       pred_transformed = sc.inverse_transform(pred)
       #y_test_transformed = sc.inverse_transform(y_test)
       
       Y_test_transformed = sc.inverse_transform(y_train)
       
       predGain = (pred_transformed[0][0]-Y_test_transformed[-1][0])/Y_test_transformed[-1][0]
       #actGain  = (y_test_transformed[0][0] - Y_test_transformed[-1][0])/X_test_transformed[-1][0]
       d = dict()
       d["predGain"] = predGain
       d["actVal"]   = Y_test_transformed[-1][0]
       #d["actGain"] = actGain
       predDict[coin] = d
       
       print("Predicted Plus for " + coin +" :" + str(predGain))
       print("Actual Value for " + coin +" :" + str(Y_test_transformed[-1][0]))
   

print("predicted Values")
print(predDict)


# Nehme 5 Best, und verteile jeweils 10€ drauf.
# Mache das jeden Tag
#%%



sortArray = predDict.items()
sortArray = sorted(sortArray, key=lambda item: item[1]["predGain"])
sortArray.reverse()

data = {'coin':[], 'amount':[], 'predGain':[]}
newPortfolio = pd.DataFrame(data)
newPortfolio.set_index("coin", inplace = True)

#%% Get Value of Current Portfolio
currentTickers = client.get_all_tickers()

def getFittingPrice(sym):
    for i in currentTickers:
        if sym == i["symbol"]:
            return float(i["price"])
        
gesamtAmountUSD = 0.0
if portfolio.shape[0] < 2:
    gesamtAmountUSD = 100.0


for i in range(portfolio.shape[0]):
    symbol = portfolio.index[i] + "USDC"
    amount = portfolio.iat[i,0]
    kurs = getFittingPrice(symbol)
    print(symbol, kurs)
    zz =  float(amount) * float(kurs)
    gesamtAmountUSD = gesamtAmountUSD + zz


kBest = 6
amountPerCurrency = gesamtAmountUSD / kBest
print("Portfolio Value:", gesamtAmountUSD, " $C")

#%%Get new Portfolio by redistributing the money in the old portfolio
for i in range(kBest):
    pG = sortArray[i][1]["predGain"]
    if pG >= 0.04:
        symb = sortArray[i][0]
        index = symb[:-4]
        newPortfolio.loc[index] = sortArray[i][1]["predGain"]
        price = getFittingPrice(symb) # Price in USDC for one
        print(symb)
        print(price)
        
        newPortfolio.loc[index]["amount"] =  amountPerCurrency / float(price)


newPortfolio.to_pickle("balance.txt")


#today = datetime.datetime.now()
    



