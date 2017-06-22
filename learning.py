
#def learning_news_effect(alone, prices=0, dates=0, names=0, lreturns=0, 
#news_data=0)



def build_word2vec_model(alone, news_data=[], faulty_news=[]):
	import numpy as np
	import gensim, logging
	import pickle

	if alone:
		f = open('./Data/processed_news_data', 'rb')
		[news_data, faulty_news] = pickle.load(f)
		f.close()


	#remove stopwords - maybe keep them

	#learn word2vec cbow - gensim
	
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	allsentences = []
	for i in news_data:
		for j in i[8]:
			allsentences.append(j)
	# train word2vec on the two sentences
	model = gensim.models.Word2Vec(allsentences, size=300, min_count=10, workers=4)

	return model





def get_news_vector(alone,model, news_data=[], faulty_news=[]): 	
	import pickle
	import numpy as np

	if alone:
		f = open('./Data/processed_news_data', 'rb')
		[news_data, faulty_news] = pickle.load(f)
		f.close()


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
	dates_news = []
	dates_news.append(news_data[0][4])

	for i in range(1,len(news_data)):
		temp = news_data[i][4].tolist()
		cur_date = temp.date().isocalendar()[1]
		if cur_date == prev_date:
			aggregated_news[cur_pos,:] = np.add(aggregated_news[cur_pos,:],vec_news[i,:])
			prev_date = cur_date
		else:
			dates_news.append(news_data[i][4])
			cur_pos += 1
			aggregated_news = np.vstack((aggregated_news, vec_news[i,:]))
			prev_date = cur_date

	return [aggregated_news, dates_news]

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def gen_xy(aggregated_news,mu,dates_news,dates_SP_weekly):
	import numpy as np
	import datetime
	#mu chages over weeks
	change_mu = np.diff(mu,axis=0)

	#find matching
	x_train = np.array([])
	x_train = np.reshape(x_train, [0,np.shape(aggregated_news[1])[0]])
	y_train = np.array([])
	y_train = np.reshape(y_train, [0,np.shape(mu[1])[0]])
	for i in range(len(dates_news)):
		x_train = np.vstack((x_train,aggregated_news[i,:]))
		temp = dates_news[i].tolist()
		temp2 = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
		try:
			ind_of_week = dates_SP_weekly.index(temp2)
			y_train = np.vstack((y_train,change_mu[ind_of_week+1,:]))
		except:
			temp3 = nearest(dates_SP_weekly,temp2)
			ind_of_week = dates_SP_weekly.index(temp3)
			y_train = np.vstack((y_train,change_mu[ind_of_week+1,:]))

	return [x_train, y_train]


def rnn_model(x_train,y_train):
	from keras.models import Sequential
	from keras.layers import LSTM
	import numpy as np

	model = Sequential()

	x_train = np.reshape(x_train, x_train.shape + (1,))

	model.add(LSTM(64, input_shape=x_train.shape[1:], return_sequences=True))
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(1))

	model.compile(loss='mean_squared_error',
	              optimizer='sgd')


	model.fit(x_train, y_train, epochs=5, batch_size=32,validation_split=0.1)
	#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
	#classes = model.predict(x_test, batch_size=128)

	return model


