
#def learning_news_effect(alone, prices=0, dates=0, names=0, lreturns=0, 
#news_data=0)



def build_word2vec_model(alone, fnum,mcount,news_data=[], faulty_news=[]):
	import numpy as np
	import gensim, logging

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





def get_news_vector(alone,model, fnum, news_data=[], faulty_news=[]): 	
	import numpy as np
	from progressbar import printProgressBar

	vec_news = np.zeros([len(news_data),fnum])
	dates_news = list()
	
	#transform messages to vectors (mean)
	prog_st = 0
	l = len(news_data) 
	printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

	for i in range(len(news_data)):
		prog_st+=1
		printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)	
		if news_data[i][8] != []:
			mean_divider = 0
			for j in news_data[i][8]:
				for k in j:
					try:
						vec_news[i,:] = np.add(vec_news[i,:], model.wv[k])
						mean_divider += 1
					except:
						continue
			vec_news[i,:] = np.divide(vec_news[i,:], mean_divider)
			dates_news.append(news_data[i][4])


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
	x = np.zeros([0,np.shape(aggregated_news[1])[0]])
	x = np.reshape(x, [0,np.shape(aggregated_news[1])[0]])
	y = np.array([])
	y = np.zeros([0,np.shape(lreturns[1])[0]])
	y = np.reshape(y, [0,np.shape(lreturns[1])[0]])
	
	j = 0
	l = len(dates_news) 
	printProgressBar(j, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


	counter_av = 0
	counter_news = -1
	prev_d = np.datetime64(datetime.date(1, 1, 1))

	for i in range(len(dates_news)):
		
		temp = dates_news[i].tolist()
		cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))

		#matching SP date
		try:
			ind_of_week = dates.index(cur_d)
		except:
			temp3 = nearest(dates,cur_d)
			temp3 = min(dates, key=lambda x: abs(x - cur_d))
			ind_of_week = dates.index(temp3)


		#add news, up
		if cur_d == prev_d:
			prev_d = cur_d
			counter_av += 1
			x[counter_news,:] = np.add(x[counter_news,:], aggregated_news[i,:])
		else:
			prev_d = cur_d
			if counter_news != -1:
				x[counter_news,:] = np.divide(x[counter_news,:], counter_av)
			counter_av = 1
			counter_news += 1
			x = np.vstack((x, aggregated_news[i,:]))
			x_dates.append(dates_news[i])

			#mu for labels
			past_mu = np.mean(lreturns[(ind_of_week-n_past):ind_of_week,:],axis=0)
			future_mu = np.mean(lreturns[ind_of_week:(ind_of_week+n_forward),:],axis=0)
			
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

	split_point = int(np.floor(np.shape(x)[0]*(1-test_split)))
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
	model.add(LSTM(8, return_sequences=True))
	model.add(LSTM(4, return_sequences=True))
	model.add(LSTM(1))

	model.compile(loss='mean_squared_error',
	              optimizer='sgd')


	model.fit(x_train, y_train, epochs=ep, batch_size=bs,validation_split=val_split)
	#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
	#classes = model.predict(x_test, batch_size=128)

	return model


