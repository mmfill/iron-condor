# iron-condor
Attempt to create a machine learning model for Iron Condor investment. Because so many apporaches are tested the code is nested in Jupyter notebooks for better understanding. The goal is to find the ideal approach for the best model to predict the data as good as possible (maximise return of investment).

A blog post for this project can be found on:

Iron Condor: The iron condor is a investment strategy using four options with the same strike date. More precise, the iron condor uses two vertical spreads, one put spread and a call spread. The put spread consists at of a short put option (P_s) with a strike price below the actual stock price and a long put option (P_l) with a strike price below the short put option P_s. The call spread consists of a short call option (C_s) above the actual stock price and a long call option (C_l) above the short call option C_s. All options have the same strike date and the spread on both legs can vary but is chosen mostly the same.

About the project:
This project is not straightforward, but four different approaches are tested for two different target data with two different models. The models used are:
- feedforward model
- LSTM model

Target data
- stock price
- short put and short call option spread

The different approaches are:
- bunch forecast
- rolling forecast
- rolling fit and rolling forecast
- stationary time series

Eight Jupyter notebooks are uploaded:
- v04 - v07 use stock data as target data
- v04 - v07 use a different approach each
- v04 - v07 apply this approach to both models
- v08 - v11 use short call and short put spreads as target data
- v08 - v11 use a different approach each
- v08 - v11 apply this approach to both models

The data used for this project are 3 csv files:
- GOOG.csv: stock price data of google over more than 10 years
- Nasdaq2.csv: Data of Nasdaq over more than 10 years
- S&P.csv: Data of S&P500 over more than 10 years


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
