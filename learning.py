
#def learning_news_effect(alone, prices=0, dates=0, names=0, lreturns=0, 
#news_data=0)

import tensorflow as tf
import numpy as np

import pickle

#if alone:
f = open('./Data/processed_data', 'rb')
[prices, dates, names, lreturns, news_data, faulty_news] = pickle.load(f)
f.close()


