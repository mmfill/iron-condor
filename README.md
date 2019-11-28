Machine learning approach for the Iron Condor

Attempt to create a machine learning model for Iron Condor investment.  The iron condor is a investment strategy using four different stock options with the same strike date. More precise, the iron condor uses two vertical spreads, one vertical put spread and a vertical call spread. The vertical put spread consists at of a short put option (P_s) with a strike price below the actual stock price and a long put option (P_l) with a strike price below the short put option P_s. The vertical call spread consists of a short call option (C_s) above the actual stock price and a long call option (C_l) above the short call option C_s. All stock options have the same strike date. The goal is to specify the strike prices of both short options. To achieve this a feedforward model and a LSTM model as well as four different apporaches are tested. For better understanding the code is nested in Jupyter notebooks. As target value also the stock price at strike date is used.

A blog post for this project can be found on: https://medium.com/@matthias.fill/a-machine-learning-approach-to-the-iron-condor-8bf1ead4af5a
Github repo: https://github.com/mmfill/iron-condor

About the project: This project is not straightforward, but four different approaches are tested for two different target data with two different models. 

The models used are:
- feedforward (FF) model
- LSTM model

Target data
-stock price
-short put and short call option spread

The different approaches are:
-bunch forecast and bunch fit
-rolling forecast and bunch fit
-rolling fit and rolling forecast
-stationary time series

Results: 
For a given stock (Google) and a fixed strike date (45 days) the stock price seems to be easier to predict and LSTM model shows better accuracy than FF model. The ideal approach would be a rolling fit/rolling forecast if run time is not a major issue.
The short option spreads are hard to predict and both model oscillate about an average value. No model shows an accurate prediction. Option spreads are therefore not a good target value. This is also backed by the fact that all approaches show similar accuracies. No approach can predict the data well.
For a further improvement model parameters of the LSTM model should be optimized.

11 files are uploaded.
Eight are Jupyter notebooks:
Iron_Condor_v04_stock-price_bunch-forecast.ipynb: Models: FF and LSTM, target data: stock price, bunch fit, bunch forecast
Iron_Condor_v05_stock-price_rolling-forecast.ipynb: Models: FF and LSTM, target data: stock price, bunch fit, rolling forecast
Iron_Condor_v06_stock-price_rolling-fit -and-forecast.ipynb: Models: FF and LSTM, target data: stock price, rolling fit, rolling forecast
Iron_Condor_v07_stock-price_rolling-forecast_stationary-data.ipynb: Models: FF and LSTM, target data: stock price, bunch fit, bunch forecast, stationary time series
Iron_Condor_v08_option-spread_bunch-forecast.ipynb: Models: FF and LSTM, target data: option spread, bunch fit, bunch forecast
Iron_Condor_v09_option-spread_rolling-forecast.ipynb: Models: FF and LSTM, target data: option spread, bunch fit, rolling forecast
Iron_Condor_v10_option-spread_rolling-fit -and-forecast.ipynb: Models:  Models: FF and LSTM, target data: option spread, rolling fit, rolling forecast
Iron_Condor_v11_option-spread_rolling-forecast_stationary-data.ipynb: Models: FF and LSTM, target data: option spread, bunch fit, bunch forecast, stationary time series

The data used for this project are in 3 csv files:
GOOG.csv: stock price data of google over more than 10 years
Nasdaq2.csv: Data of Nasdaq over more than 10 years
S&P.csv: Data of S&P500 over more than 10 years

Python 3 is used 

Packages needed: 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm 
import numpy as np 
import scipy.stats as si 
from IPython.display import Image 
import datetime 
import time 
from keras.models import Sequential 
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers import LSTM 
from sklearn import preprocessing, metrics
