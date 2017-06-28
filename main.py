from data_loading import load_news_data, load_SP_data 
from learning import  gen_dict, gen_xy_daily, train_test_split
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

#mean change approach
n_forward_list = 3
n_past_list = 60



#load and preprocess data
print(str(datetime.datetime.now())+': Start reading in news:')
[news_data, faulty_news] = load_news_data(path_to_news_files,True)
print(str(datetime.datetime.now())+': Successfully read all news')


#keras tokenizer experiment

 


print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[prices, dates, names, lreturns] = load_SP_data(stocks_used, firms_used)
print(str(datetime.datetime.now())+': Successfully read all data')


print(str(datetime.datetime.now())+': Start generating x and y')
#dict_words = gen_dict(news_data)
[x,y] = gen_xy_daily(news_data,lreturns,dates, vocab_size)
print(str(datetime.datetime.now())+': Successfully generated x and y')


x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)

#model
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, Flatten, Convolution1D, MaxPooling1D, LSTM

model = Sequential([
	Embedding(vocab_size+1, 40, input_length=np.shape(x_train)[1]),
	Flatten(),
	Dense(100,activation="relu"),
	Dropout(0.7),
	Dense(1,activation="sigmoid")
	])


conv_model = Sequential([
	Embedding(vocab_size+1, 32, input_length=np.shape(x_train)[1]),
	Dropout(0.2),
	Convolution1D(64, 5, border_mode='same', activation="relu" ),
	Dropout(0.2),
	MaxPooling1D(),
	Flatten(),
	Dense(100,activation="relu"),
	Dropout(0.7),
	Dense(1,activation="sigmoid")
	])

rnn_model = Sequential([
	Embedding(vocab_size+1, 32, input_length=np.shape(x_train)[1]),
	LSTM(52,return_sequences=True),
	LSTM(1)
	])

rnn_model.compile(loss='mean_squared_error', optimizer='sgd')
rnn_model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=5,batch_size=25)

model.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])
model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=5,batch_size=10)

conv_model.compile(loss="mean_squared_error",optimizer="Adam")
conv_model.fit(x_train,y_train[:,0],validation_data=(x_test,y_test[:,0]),epochs=2,batch_size=25)


#evaluate
plot_pred_true(y_test,rnn_model.predict(x_test, batch_size=25))

