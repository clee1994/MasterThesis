
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

#get data and matching labels
temp = news_data[0][4].tolist()
prev_date = temp.date().isocalendar()[1]
aggregated_news = np.array([vec_news[0,:]])
cur_pos = 0

for i in range(1,len(news_data)):
	temp = news_data[i][4].tolist()
	cur_date = temp.date().isocalendar()[1]
	if cur_date == prev_date:
		aggregated_news[cur_pos,:] = np.add(aggregated_news[cur_pos,:],vec_news[i,:])
	else:
		cur_pos += 1
		aggregated_news = np.vstack((aggregated_news, vec_news[i,:]))




#RNN to find vector effect on mean/variance
from keras.models import Sequential

model = Sequential()

from keras.layers import LSTM

x_train = np.reshape(x_train, x_train.shape + (1,))

model.add(LSTM(64, input_dim=100, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(1))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)



#--------------------------------------------------------

x_train = np.random.rand(300,100)
y_train = np.random.rand(300,1)
from keras.models import Sequential
from keras import losses, optimizers

model = Sequential()

from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=1))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=5, batch_size=32)




#RNN to find vector effect on mean/variance
from keras.models import Sequential

model = Sequential()

from keras.layers import SimpleRNN

x_train = np.random.rand(300,10)
y_train = np.sum(x_train, axis=1)

x_test = np.random.rand(300,10)
y_test = np.sum(x_test, axis=1)

x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))

model.add(SimpleRNN(64, input_shape=x_train.shape[1:], return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(1))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

#random -> gen


model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)

#return word2vec and coefficients

