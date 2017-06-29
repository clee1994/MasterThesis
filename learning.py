
def gen_xy_daily(news,lreturns,dates_stocks,features,window,mcount,ht):
	import datetime
	import numpy as np
	from progressbar import printProgressBar
	from gensim.models.doc2vec import TaggedDocument
	from gensim.models import Doc2Vec

	#variables
	documents = []
	y = []
	x = []
	words = []

	#progressbar
	prog_st = 0
	l = len(news) 
	printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


	prev_d = np.datetime64(datetime.date(1, 1, 1))
	for i in news:
		prog_st += 1
		printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


		temp_d = []
		temp = i[4].tolist()
		cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))

		#y
		if cur_d == prev_d:
			prev_d = cur_d
		else:
			#text/x
			if prev_d != np.datetime64(datetime.date(1, 1, 1)):
				documents.append(TaggedDocument(words,str(cur_d)))
				words = []
			prev_d = cur_d
			#mu/y -> what do I want, the mean next day, average next three days
			try:
				#temp_mu = np.nanmean(lreturns[list(dates_stocks).index(cur_d):(list(dates_stocks).index(cur_d)+3),:],axis=0)
				temp_mu = np.sign(lreturns[list(dates_stocks).index(cur_d),:])
				#temp_mu = lreturns[list(dates_stocks).index(cur_d),:]
				y.append(temp_mu)
			except:
				ind_temp = min(dates_stocks, key=lambda x: abs(x - cur_d))
				#temp_mu = lreturns[list(dates_stocks).index(ind_temp),:]
				#temp_mu = np.nanmean(lreturns[list(dates_stocks).index(ind_temp):(list(dates_stocks).index(ind_temp)+3),:],axis=0)
				temp_mu = np.sign(lreturns[list(dates_stocks).index(ind_temp),:])
				y.append(temp_mu)

		#x and skip last sentence
		for j in range(len(i[ht])-1):
			for hj in i[ht][j]:
				 words.append(hj)
	
	try:
		documents.append(TaggedDocument(words,tags=str(cur_d)))
	except:
		pass

	model = Doc2Vec(size=features, window=window, min_count=mcount, workers=4)

	model.build_vocab(documents)

	model.train(documents,total_examples=model.corpus_count, epochs=model.iter)


	for i in documents:
		x.append(model.infer_vector(i[0]))

	return [np.array(x), np.array(y)]






def train_test_split(x,y,test_split):
	import numpy as np
	split_point = int(np.floor(np.shape(x)[0]*(1-test_split)))
	x_train = x[0:split_point,:]
	y_train = y[0:split_point,:]
	x_test = x[(split_point+1):,:]
	y_test = y[(split_point+1):,:]
	return x_train,y_train,x_test,y_test






