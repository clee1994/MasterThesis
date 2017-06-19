
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
 
allsentences = []
for i in news_data:
	for j in i[8]:
		allsentences.append(j)
# train word2vec on the two sentences
model = gensim.models.Word2Vec(allsentences, size=300, min_count=10, workers=4)

vec_news = np.zeros([len(news_data),len(model.wv['and'])])
#transform messages to vectors (mean/sum)
for i in range(len(news_data)):
	for j in news_data[i][8]:
		for k in j:
			try:
				vec_news[i,:] = np.add(vec_news[i,:], model.wv[k])
			except:
				continue


#RNN to find vector effect on mean/variance
from keras.models import Sequential

model = Sequential()

from keras.layers import LSTM

model.add(LSTM(64, input_dim=64, input_length=300, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)

#return word2vec and coefficients

