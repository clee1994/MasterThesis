
#def learning_news_effect(alone, prices=0, dates=0, names=0, lreturns=0, 
#news_data=0)

import tensorflow as tf
import numpy as np

import pickle

#if alone:
f = open('./Data/processed_data', 'rb')
[prices, dates, names, lreturns, news_data, faulty_news, mu, sigma] = pickle.load(f)
f.close()


#remove stopwords - maybe keep them

#learn word2vec cbow - gensim
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

#transform messages to vectors (mean/sum)

#RNN to find vector effect on mean/variance

#return word2vec and coefficients

