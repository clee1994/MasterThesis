
#def learning_news_effect(alone, prices=0, dates=0, names=0, lreturns=0, 
#news_data=0)



def build_word2vec_model(alone, fnum,mcount,news_data=[], faulty_news=[]):
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
	model = gensim.models.Word2Vec(allsentences, size=fnum, min_count=mcount, workers=8)

	return model





def get_news_vector(alone,model, news_data=[], faulty_news=[]): 	
	import pickle
	import numpy as np
	from progressbar import printProgressBar

	if alone:
		f = open('./Data/processed_news_data', 'rb')
		[news_data, faulty_news] = pickle.load(f)
		f.close()


	vec_news = np.zeros([len(news_data),len(model.wv['and'])])
	dates_news = list()
	#transform messages to vectors (mean/sum)

	prog_st = 0
	l = len(news_data) 
	printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


	for i in range(len(news_data)):
		prog_st+=1
		printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
		for j in news_data[i][8]:
			for k in j:
				try:
					vec_news[i,:] = np.add(vec_news[i,:], model.wv[k])
				except:
					continue
		dates_news.append(news_data[i][4])

	#get data and matching labels
	# temp = news_data[0][4].tolist()
	# prev_date = temp.date().isocalendar()[1]
	# aggregated_news = np.array([vec_news[0,:]])
	# cur_pos = 0
	# dates_news = []
	# dates_news.append(news_data[0][4])

	# for i in range(1,len(news_data)):
	# 	temp = news_data[i][4].tolist()
	# 	cur_date = temp.date().isocalendar()[1]
	# 	if cur_date == prev_date:
	# 		aggregated_news[cur_pos,:] = np.add(aggregated_news[cur_pos,:],vec_news[i,:])
	# 		prev_date = cur_date
	# 	else:
	# 		dates_news.append(news_data[i][4])
	# 		cur_pos += 1
	# 		aggregated_news = np.vstack((aggregated_news, vec_news[i,:]))
	# 		prev_date = cur_date
	dates_news = np.array(dates_news)

	return [vec_news, dates_news]

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def gen_xy(aggregated_news,lreturns,dates,dates_news,n_forward,n_past,mu_var,names, test_split):
	import numpy as np
	import datetime
	from progressbar import printProgressBar

	dates = list(dates)
	x_dates = list()

	#find matching
	x = np.array([])
	x = np.reshape(x, [0,np.shape(aggregated_news[1])[0]])
	y = np.array([])
	y = np.reshape(y, [0,np.shape(lreturns[1])[0]])
	
	j = 0
	l = len(dates_news) 
	printProgressBar(j, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

	for i in range(len(dates_news)):
		x = np.vstack((x, aggregated_news[i,:]))
		x_dates.append(dates_news[i])
		temp = dates_news[i].tolist()
		temp2 = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
		try:
			ind_of_week = dates.index(temp2)
		except:
			temp3 = nearest(dates,temp2)
			temp3 = min(dates, key=lambda x: abs(x - temp2))
			ind_of_week = dates.index(temp3)

		if mu_var:
			past_mu = np.mean(lreturns[(ind_of_week-n_past):ind_of_week,:],axis=0)
			future_mu = np.mean(lreturns[ind_of_week:(ind_of_week+n_forward),:],axis=0)
		else:
			past_mu = np.var(lreturns[(ind_of_week-n_past):ind_of_week,:],axis=0)
			future_mu = np.var(lreturns[ind_of_week:(ind_of_week+n_forward),:],axis=0)
		y = np.vstack((y, (future_mu-past_mu)))
		j += 1
		printProgressBar(j, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


	used_stocks = list()
	bad_stocks = list()

	#drop stocks with insufficient data
	for i in range(np.shape(y)[1]):
		if (np.sum(np.isnan(y[:,i]))/len(y[:,i])) > 0.1:
			bad_stocks.append(i)
		else:
			used_stocks.append(names[i])

	y = np.delete(y,bad_stocks,1)

	split_point = np.floor(np.shape(x)[0]*(1-test_split))
	x_train = x[0:split_point,:]
	y_train = y[0:split_point,:]
	x_test = x[(split_point+1):,:]
	x_test = np.reshape(x_test, x_test.shape + (1,))
	y_test = y[(split_point+1):,:]
	x_dates = np.array(x_dates)
	x_dates_train = x_dates[0:split_point]
	x_dates_test = x_dates[(split_point+1):]

	return [x_train, y_train, x_test, y_test, used_stocks, x_dates_train, x_dates_test]


def rnn_model(x_train,y_train, val_split, bs, ep):
	from keras.models import Sequential
	from keras.layers import LSTM
	import numpy as np

	model = Sequential()

	x_train = np.reshape(x_train, x_train.shape + (1,))


	#here comes a design questions, how many layers and how many neurons per layer
	model.add(LSTM(64, input_shape=x_train.shape[1:], return_sequences=True))
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(1))

	model.compile(loss='mean_squared_error',
	              optimizer='sgd')


	model.fit(x_train, y_train, epochs=ep, batch_size=bs,validation_split=val_split)
	#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
	#classes = model.predict(x_test, batch_size=128)

	return model


