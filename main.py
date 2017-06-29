from data_loading import load_news_data, load_SP_data 
from learning import gen_xy_daily, train_test_split
from evaluation import plot_pred_true, evaluate_portfolio
from stocks_small import stocks_used
import numpy as np
import datetime 


#modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
firms_used = 25

#traning splits
test_split = 0.15
validation_split = 0.12

#
vocab_size = 4000
feature_size = 700

#mean change approach
n_forward_list = 3
n_past_list = 60



#load and preprocess data
print(str(datetime.datetime.now())+': Start reading in news:')
[news_data, faulty_news] = load_news_data(path_to_news_files,False)
print(str(datetime.datetime.now())+': Successfully read all news')




print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[prices, dates, names, lreturns] = load_SP_data(stocks_used, firms_used)
print(str(datetime.datetime.now())+': Successfully read all data')


#pickle dump and load
import pickle
pickle.dump([news_data, prices, dates, names, lreturns], open( "small_raw", "wb" ) )

[news_data, prices, dates, names, lreturns] = pickle.load( open( "small_raw", "rb" ) )


feature_size = 990

for feature_size in [90, 150, 350, 500, 700]:

	print(str(datetime.datetime.now())+': Start generating xy:')
	[x,y] = gen_xy_daily(news_data,lreturns,dates,feature_size,8,50)
	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
	print(str(datetime.datetime.now())+': Successfully generated xy')


	import pickle
	pickle.dump([x_train, y_train, x_test, y_test], open( "small_xy_"+str(feature_size), "wb" ) )
	print(str(feature_size))



#ht: 2 headline, 8 text
print(str(datetime.datetime.now())+': Start generating xy:')
[x,y] = gen_xy_daily(news_data,lreturns,dates,150,7,10,8)
x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
print(str(datetime.datetime.now())+': Successfully generated xy')




#ATTENTION!!!!
[x_train, y_train, x_test, y_test] = pickle.load( open( "small_xy_500", "rb" ) )

#ridge regression easy model
from sklearn.linear_model import Ridge
from sklearn import svm

#3000 standard
clf = Ridge(alpha=50)
clf.fit(x_train, y_train)


plot_pred_true(y_test,clf.predict(x_test))

import tensorflow as tf
from keras.losses import mean_squared_error
sess = tf.InteractiveSession()
np.mean(mean_squared_error(y_test,clf.predict(x_test)).eval())




#classification
y_train[y_train < 0] = 0
y_test[y_test < 0] = 0
clf = svm.SVC()
clf.fit(x_train, y_train)
res =  np.reshape(np.array(clf.predict(x_test)),[387,1])
np.sum(np.abs(np.subtract(y_test,res)))/np.shape(y_test)[0]


#-------
# create model
model = Sequential()
model.add(Dense(13, input_dim=x_train.shape[1:], kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)
plot_pred_true(y_test,model.predict(x_test, batch_size=25))






x_train = np.transpose(x_train)

#single layer neuronal net
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, Flatten, Convolution1D, MaxPooling1D, LSTM

model = Sequential([
	#Embedding(vocab_size+1, 40, input_length=np.shape(x_train)[1]),
	Dense(100,input_shape=x_train.shape[1:], activation="relu"),
	#Flatten(),
	Dropout(0.7),
	Dense(1,activation="relu"),
	])
model.compile(loss="mean_squared_error",optimizer="Adam")
model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)
plot_pred_true(y_test,model.predict(x_test))




#LSTM
rnn_model = Sequential([
	LSTM(52,input_shape=x_train.shape[1:],return_sequences=True),
	LSTM(36, return_sequences=True),
	LSTM(1)
	])
rnn_model.compile(loss="mean_squared_error",optimizer="Adam")

x_train_l = np.reshape(x_train, x_train.shape + (1,))
x_test_l = np.reshape(x_test, x_test.shape + (1,))

rnn_model.fit(x_train_l,y_train[:,0],validation_data=(x_test_l,y_test[:,0]),epochs=2,batch_size=25)
plot_pred_true(y_test,rnn_model.predict(x_test_l))

rnn2_model = Sequential()

rnn2_model.add(LSTM(52, input_shape=x_train.shape[1:], return_sequences=True))
rnn2_model.add(LSTM(36, return_sequences=True))
#model.add(LSTM(4, return_sequences=True))
rnn2_model.add(LSTM(1))


from keras import backend as K
K.clear_session()

#CNN
cnn_model = Sequential([
	Convolution1D(64, 5, border_mode='same',input_shape=x_train.shape[1:], activation="relu" ),
	Dropout(0.2),
	MaxPooling1D(),
	Flatten(),
	Dense(100,activation="relu"),
	Dropout(0.7),
	Dense(1)
	])
cnn_model.compile(loss="mean_squared_error",optimizer="Adam")
cnn_model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)
plot_pred_true(y_test,cnn_model.predict(x_test))




#--------------------------------------------------------


x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))

model.compile(loss="mean_squared_error",optimizer="Adam")
model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)



#model
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, Flatten, Convolution1D, MaxPooling1D, LSTM


model = Sequential([
	#Embedding(vocab_size+1, 40, input_length=np.shape(x_train)[1]),
	Dense(32,input_shape=x_train.shape[1:], activation="relu"),
	#Flatten(),
	Dense(100,activation="relu"),
	Dropout(0.7),
	Dense(1)
	])

conv_model = Sequential([
	#Embedding(vocab_size+1, 32, input_length=np.shape(x_train)[1]),
	#Dropout(0.2),
	Convolution1D(64, 5, border_mode='same', activation="relu" ),
	Dropout(0.2),
	MaxPooling1D(),
	Flatten(),
	Dense(100,activation="relu"),
	Dropout(0.7),
	Dense(1)
	])

rnn_model = Sequential([
	#Embedding(vocab_size+1, 32, input_length=np.shape(x_train)[1]),
	LSTM(52,return_sequences=True),
	LSTM(36, return_sequences=True),
	LSTM(1)
	])

rnn_model.compile(loss='mean_squared_error', optimizer='Adam')
rnn_model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)

model.compile(loss="mean_squared_error",optimizer="Adam")
model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)

conv_model.compile(loss="mean_squared_error",optimizer="Adam")
conv_model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)


#evaluate
plot_pred_true(y_test,model.predict(x_test, batch_size=25))
plot_pred_true(y_test,conv_model.predict(x_test, batch_size=25))
plot_pred_true(y_test,rnn_model.predict(x_test, batch_size=25))




