


def data_label_method_sign(lreturns, cur_d,dates_stocks):
	import numpy as np
	try:
		ret_val = np.sign(lreturns[list(dates_stocks).index(cur_d)+1,:])
	except:
		ind_temp = min(dates_stocks, key=lambda x: abs(x - cur_d))
		ret_val = np.sign(lreturns[list(dates_stocks).index(ind_temp)+1,:])
	return ret_val

def data_label_method_val(lreturns, cur_d,dates_stocks):
	import numpy as np
	try:
		ret_val = lreturns[list(dates_stocks).index(cur_d)+1,:]
	except:
		ind_temp = min(dates_stocks, key=lambda x: abs(x - cur_d))
		ret_val = lreturns[list(dates_stocks).index(ind_temp)+1,:]
	return ret_val

#now I am taking past and future to correctly estimate var/cov
def data_label_method_cov(lreturns, cur_d,dates_stocks):
	#covariance future/past... important!!
	import numpy as np

	#important
	n = 4

	try:
		indt = list(dates_stocks).index(cur_d)
		#ret_val = np.cov(lreturns[indt:indt+3,0],lreturns[indt:indt+3,1])[0,1]
	except:
		ind_temp = min(dates_stocks, key=lambda x: abs(x - cur_d))
		indt = list(dates_stocks).index(ind_temp)
	try:
		ret_val = np.cov(lreturns[0,indt-n:indt+n],lreturns[1,indt-n:indt+n])[0,1]
	except:
		ret_val = np.cov(lreturns[0,indt-n:],lreturns[1,indt-n:])[0,1]
	return ret_val

def gen_xy_daily(news,lreturns,dates_stocks,features,window,mcount,ylm,dm_opt, tables=False, dmm=0, dmc=0):
	import datetime
	import numpy as np
	from gensim.models.doc2vec import TaggedDocument
	from gensim.models import Doc2Vec

	#variables
	documents = []
	y = []
	x = []
	words = []
	x_dates = []
	doc_c = 0

	
	print('Progress: [', end='', flush=True)
	prog_c = 0
	bp = 0

	prev_d = np.datetime64(datetime.date(1, 1, 1))
	for i in news:
		prog_c += 1
		if bp < np.sum((prog_c/len(news)) >  np.linspace(1/30,1,30)):
			print('=', end='', flush=True)
			bp += 1

		temp_d = []

		#temp = i[4].tolist()
		cur_d = np.datetime64(i[0])

		#y
		if cur_d == prev_d:
			prev_d = cur_d
		else:
			#text/x
			if prev_d != np.datetime64(datetime.date(1, 1, 1)):
				documents.append(TaggedDocument(words,str(doc_c)))
				doc_c = doc_c + 1
				words = []
			x_dates.append(cur_d)
			prev_d = cur_d
			#mu/y -> what do I want, the mean next day, average next three days
			y.append(ylm(lreturns, cur_d,dates_stocks))

		#x and skip last sentence / headlines... no last sentence only one
		for hj in i[1]:
			words.append(hj)
	
	try:
		documents.append(TaggedDocument(words,tags=str(doc_c)))
		#x_dates.append(cur_d)
	except:
		pass

	print('] Done')

	#maybe also dm = 0 -> different methode
	#iter -> number of epochs 
	#way more models then here -> think about extending
	model = Doc2Vec(dm = dm_opt, size=features, window=window, min_count=mcount, workers=4, dbow_words=1,dm_mean=dmm,dm_concat=dmc)

	#tag goes into vocab too.... reconsider
	model.build_vocab(documents)

	model.train(documents,total_examples=model.corpus_count, epochs=40)

	for i in documents:
		x.append(model.infer_vector(i[0]))


	#evaluation
	
	if tables:

		doc_id = np.random.randint(model.docvecs.count)
		sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
		target = ' '.join(documents[doc_id].words)
		closest = ' '.join(documents[int(sims[0][0])].words)
		least = ' '.join(documents[int(sims[len(sims) - 1][0])].words)

		chars_pl = 65



		f = open('Output/tables/'+str(datetime.datetime.now())+'target.tex', 'w')
		f.write('"'+ target[0:(chars_pl-1)] + '\n')

		for i in range(9):
			f.write(target[(i+1)*(chars_pl-1):(i+2)*(chars_pl-1)] + '\n')
		f.write('... \n')

		for i in np.arange(9,0,-1):
			f.write(target[-(i+2)*chars_pl:-(i+1)*chars_pl] + '\n')
		f.write(target[-chars_pl:-1]+'"\n')
		f.write('Date: '+ str(x_dates[doc_id].astype('M8[D]')) + '\n')
		f.write('Number of characters: ' + str(len(target)) + '\n')
		f.close() 


		f = open('Output/tables/'+str(datetime.datetime.now())+'closest.tex', 'w')
		f.write('"'+ closest[0:(chars_pl-1)]+ '\n')

		for i in range(9):
			f.write(closest[(i+1)*(chars_pl-1):(i+2)*(chars_pl-1)] + '\n')
		f.write('... \n')

		for i in np.arange(9,0,-1):
			f.write(closest[-(i+2)*chars_pl:-(i+1)*chars_pl] + '\n')
		f.write(closest[-chars_pl:-1]+'"\n')
		f.write('Date: '+ str(x_dates[int(sims[0][0])].astype('M8[D]')) + '\n')
		f.write('Number of characters: ' + str(len(closest)) + '\n')
		f.close() 

		f = open('Output/tables/'+str(datetime.datetime.now())+'least.tex', 'w')
		f.write('"'+ least[0:(chars_pl-1)]+ '\n')

		for i in range(9):
			f.write(least[(i+1)*(chars_pl-1):(i+2)*(chars_pl-1)] + '\n')
		f.write('... \n')

		for i in np.arange(9,0,-1):
			f.write(least[-(i+2)*chars_pl:-(i+1)*chars_pl] + '\n')
		f.write(least[-chars_pl:-1]+'"\n')
		f.write('Date: '+ str(x_dates[int(sims[len(sims) - 1][0])].astype('M8[D]')) + '\n')
		f.write('Number of characters: ' + str(len(least)) + '\n')
		f.close()

		#words -> actually not of relevance but cool to see
		import random
		exword = random.choice(model.wv.index2word)
		similars_words = str(model.most_similar(exword, topn=20)).replace('), ',')\n')

		f = open('Output/tables/'+str(datetime.datetime.now())+'wordss.tex', 'w')
		f.write('"'+exword + '"\n')
		f.write(similars_words)
		f.close() 

	return [np.array(x), np.array(y), np.array(x_dates)]



def train_test_split(x,y,test_split):
	import numpy as np

	split_point = int(np.floor(np.shape(x)[0]*(1-test_split)))
	if y.ndim < 2:
		y_train = y[0:split_point]
		y_test = y[(split_point):]
	else:
		y_train = y[0:split_point,:]
		y_test = y[(split_point):,:]

	x_train = x[0:split_point,:]
	x_test = x[(split_point):,:]
	
	return x_train,y_train,x_test,y_test


#maybe benchmark also based on dates
def bench_mark_mu(lreturns,dates_stocks,n_past,len_o):
	import numpy as np
	ret_val = []
	for i in range(len(lreturns)-(n_past+1)):
		ret_val.append(np.mean(lreturns[i:(i+n_past),:],axis=0))

	ret_val = np.array(ret_val)
	ind_len = np.shape(ret_val)[0]-1
	return ret_val[ind_len-len_o:ind_len]



def bench_mark_cov(lreturns,dates_stocks,n_past,len_o):
	import numpy as np
	ret_val = []
	for i in range(np.shape(lreturns)[1]-(n_past+1)):
		#try:

		ret_val.append(np.cov(lreturns[0,i:(i+n_past)],lreturns[1,i:(i+n_past)])[0,1])
		#except:
		#	ret_val.append(np.cov(lreturns[0,i:],lreturns[1,i:])[0,1])


	ret_val = np.array(ret_val)
	ind_len = np.shape(ret_val)[0]-1
	return ret_val[ind_len-len_o:ind_len]





def calibrate_doc2vec(lreturns,dates,test_split,news_data):
	from sklearn import svm
	import numpy as np

	#default values
	ws_def = 8
	mc_def = 10
	dm_def = 1
	dmm_def=0 
	dmc_def=0

	#parameter space for gensim 
	fts_space = np.linspace(150,900,4,dtype=int)
	ws_space = np.linspace(2,25,4,dtype=int)
	mc_space = np.linspace(0,50,4,dtype=int)
	x_dm_space = [0,1]
	x_dmm_space = [0,1]
	x_dcm_space = [0,1]

	loss = []
	x_val = []
	for fts in fts_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts,ws_def,mc_def,data_label_method_sign, dm_def, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
	fts_opt = fts_space[np.argmax(loss)]

	loss = []
	x_val = []
	for ws in ws_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws,mc_def,data_label_method_sign, dm_def, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
	ws_opt = ws_space[np.argmax(loss)]

	loss = []
	x_val = []
	for mc in mc_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc,data_label_method_sign, dm_def, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
	mc_opt = mc_space[np.argmax(loss)]

	loss = []
	x_val = []
	for dm in x_dm_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc_opt,data_label_method_sign, dm, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
	dm_opt = x_dm_space[np.argmax(loss)]

	loss = []
	x_val = []
	for dmm in x_dmm_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc_opt,data_label_method_sign, dm_opt, False, dmm, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
	dmm_opt = x_dmm_space[np.argmax(loss)]

	loss = []
	x_val = []
	for dmc in x_dcm_space:
		[x,y,dates] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc_opt,data_label_method_sign, dm_opt, False, dmm_opt, dmc)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
	dmc_opt = x_dcm_space[np.argmax(loss)]

	return np.matrix(x_val[np.argmax(loss)]), dates


		
def make_pred_sort_table(firm_ind_u, loss, names):
	import numpy as np

	f = open('Output/tables/pred_sort.tex', 'w')
	f.write('\\begin{tabular}{ r | l }\n')
	f.write('Stock Ticker & MSE \\\\ \n ')
	f.write('\hline \n')
	for i in range(10):
		f.write(names[firm_ind_u[i]]+' & '+ "{:.4f}".format((loss[i]))+' \\\\ \n ')
	f.write('\hline \n')
	f.write('\hline \n')
	f.write('Mean & '+ "{:.4f}".format(np.nanmean(loss[:]))+' \\\\ \n ')
	f.write('\\end{tabular}')
	f.close() 


def sort_predictability(news_data,lreturns,dates,test_split,names):
	import datetime
	import numpy as np
	from sklearn.linear_model import LinearRegression
	from sklearn.model_selection import cross_val_score

	#hard coded values... review -> definitly different
	[x,y,_] = gen_xy_daily(news_data,lreturns,dates,340,8,21,data_label_method_sign,1)

	loss_ar_svm = []

	for i in range(np.shape(y)[1]):
		clf = LinearRegression(n_jobs=-1)
		ind_mask = np.invert(np.isnan(y[:,i]))

		if np.sum(ind_mask) > 0:
			scores = cross_val_score(clf, x[ind_mask,:], y[ind_mask,i], cv=5, n_jobs = -1, scoring='neg_mean_squared_error')
			loss_ar_svm.append(scores.mean())
		else:
			loss_ar_svm.append(-np.inf)

	npal = np.multiply(np.array(loss_ar_svm),-1)
	firm_ind_u = np.argsort(npal[npal!=1])

	make_pred_sort_table(firm_ind_u, npal[np.argsort(npal[npal!=1])], names)

	return firm_ind_u



def produce_y_ret(returns,dates_prices,dates_news):
	import numpy as np
	import datetime

	dates_prices = dates_prices.astype(datetime.date)
	dates_news = list(map(datetime.datetime.date, dates_news.astype(datetime.date)))

	y = []
	for i in dates_news:
		try:
			ret_val = returns[list(dates_prices).index(i)+1]
		except:
			ind_temp = min(dates_prices, key=lambda x: abs(x - i))
			ret_val = returns[list(dates_prices).index(ind_temp)+1]
		y.append(ret_val)

	return np.array(y).ravel()

def produce_y_cov(returns,dates_prices,dates_news):
	import numpy as np
	import datetime

	n = 3

	dates_prices = dates_prices.astype(datetime.date)
	dates_news = list(map(datetime.datetime.date, dates_news.astype(datetime.date)))

	y = []
	for i in dates_news:
		try:
			indt = list(dates_prices).index(i)
		except:
			ind_temp = min(dates_prices, key=lambda x: abs(x - i))
			indt = list(dates_prices).index(ind_temp)
		try:
			ret_val = np.cov(returns[0,indt-n:indt+n],returns[1,indt-n:indt+n])[0,1]
		except:
			ret_val = np.cov(returns[0,indt-n:],returns[1,indt-n:])[0,1]

		y.append(ret_val)

	return np.array(y).ravel()



def produce_mu_cov(x, test_split, lreturns, dates_prices, dates_news, n_past, names, firm_ind_u,reg_method):
	import numpy as np

	show_p = False
	stables = False
	# 5. single stock parameter calibration & get improved mu estimates
	mu_p_ts = np.empty((int(np.ceil(np.shape(x)[0]*test_split)),0), float)
	for i in firm_ind_u:
		if i == firm_ind_u[0]:
			stables = True
		temp1 = np.transpose(np.matrix( lreturns[:,i]))
		y = produce_y_ret(temp1,dates_prices, dates_news)

		mu_p_ts = np.concatenate([mu_p_ts,np.reshape(reg_method(x, y, test_split, temp1, dates_prices, n_past,i,bench_mark_mu, "Mean",names[i],show_p,stables),(np.shape(mu_p_ts)[0],1))],axis=1)
		print(str(datetime.datetime.now())+': Successfully produced mu_p_ts for '+names[i])
		if i == firm_ind_u[0]:
			stables = False


	# 7. single stock parameter calibration & get improved cov estimates
	cov_p_ts = np.zeros([int(np.ceil(np.shape(x)[0]*test_split)),len(firm_ind_u),len(firm_ind_u)])
	for i in range(len(firm_ind_u)):
		for j in range(i+1):
			if (i == j) and (i == 0):
				stables = True
			if (i == 0) and (j == 1):
				stables = True
			temp1 = np.transpose(np.matrix( lreturns[:,[i,j]]))
			y = produce_y_cov(temp1,dates_prices, dates_news)
			if i == j:
				label_text = "Variance"
				l2_test = names[i]
			else:
				label_text = "Covariance"
				l2_test = names[i] + " and " + names[j]
			cov_p_ts[:,i,j] = reg_method(x, y, test_split, temp1, dates_prices, n_past,i,bench_mark_cov, label_text, l2_test ,show_p,stables)
			cov_p_ts[:,j,i] = cov_p_ts[:,i,j]
			print(str(datetime.datetime.now())+': Successfully produced co_p_ts for '+names[firm_ind_u[i]]+' and '+names[firm_ind_u[j]])
			if (i == j) and (i == 0):
				stables = False
			if (i == 0) and (j == 1):
				stables = False
	return mu_p_ts, cov_p_ts


#TFIDF and n-grams
def tfidf_vector(n_gram, news_data, lreturns, dates_stocks, test_split):
	import numpy as np
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import CountVectorizer
	import datetime

	corpus = []
	temp_word = ""
	x_dates = []
	prev_d = np.datetime64(datetime.date(1, 1, 1))

	for i in news_data:
		cur_d = np.datetime64(i[0])

		#y
		if cur_d == prev_d:
			prev_d = cur_d
		else:
			if prev_d != np.datetime64(datetime.date(1, 1, 1)):
				#documents.append(TaggedDocument(words,str(doc_c)))
				#doc_c = doc_c + 1
				corpus.append(temp_word)
				temp_word = ""
			x_dates.append(cur_d)
			prev_d = cur_d
		for hj in i[1]:
			temp_word = temp_word + hj + " "
			#words.append(hj)
	corpus.append(temp_word)

	
	fts_space = np.linspace(900,4000,10,dtype=int)

	lossC = []
	lossT = []
	x_C = []
	x_T = []
	for i in fts_space:
		vectorizerCount = CountVectorizer(analyzer='word',ngram_range=(1,n_gram),stop_words='english',min_df=3,max_features=i)
		x = vectorizerCount.fit_transform(corpus).toarray()
		y = np.array([data_label_method_val(lreturns, cur_d, dates_stocks) for cur_d in x_dates ])
		lossC.append(val_cv_eval(x,y,test_split))
		x_C.append(x)


		vectorizerTFIDF = TfidfVectorizer(analyzer='word',ngram_range=(1,n_gram),stop_words='english',min_df=3,max_features=i)
		x = vectorizerTFIDF.fit_transform(corpus).toarray()
		y = np.array([data_label_method_val(lreturns, cur_d,dates_stocks) for cur_d in x_dates ])
		lossT.append(val_cv_eval(x,y,test_split))
		x_T.append(x)

	return x_C[np.argmax(lossC)], x_T[np.argmax(lossT)], x_dates

def val_cv_eval(x,y,split):
	from sklearn.model_selection import cross_val_score
	from sklearn.linear_model import LinearRegression 
	import numpy as np

	ind_mask = np.invert(np.isnan(y)).ravel()

	clf = LinearRegression(n_jobs = -1)
	if np.sum(ind_mask) > 0:
		#different score neg_mean_squared_error / r2 
		scores = cross_val_score(clf, x[ind_mask,:], y[ind_mask], cv=5, n_jobs = -1, scoring='neg_mean_squared_error')
		return scores.mean()
	else:
		return 0

def estimate_linear(x_cal, y_cal, test_split, lreturns, dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from sklearn.linear_model import LinearRegression
	import numpy as np
	from evaluation import plot_pred_true_b, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)


	#benchmark estimates -> this one is corrupt!
	bench_y = benchm_f(lreturns,dates,n_past,len(y_test))


	#1. model selection with cross validation and grid search
	model = LinearRegression(n_jobs = -1)
	
	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	model = model.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])
	

	#2. produce estimates
	mu_p_ts = model.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	return mu_p_ts

def estimate_ridge(x_cal, y_cal, test_split, lreturns, dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from sklearn.kernel_ridge import KernelRidge
	from sklearn.model_selection import GridSearchCV
	import numpy as np
	from evaluation import plot_pred_true_b, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)


	#benchmark estimates -> this one is corrupt!
	bench_y = benchm_f(lreturns,dates,n_past,len(y_test))


	#1. model selection with cross validation and grid search
	alpha_range1 = np.geomspace(0.1,80, 12)
	#alpha_range = np.geomspace(1e-8,40, 12)
	gamma_range = np.geomspace(1e-2,12,10)
	#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise -> kernels
	RR_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'alpha': alpha_range1},
					{'kernel': ['linear'], 'alpha': alpha_range1}]

	RR_model = KernelRidge(alpha=30)
	clf = GridSearchCV(RR_model, RR_parameters,scoring='neg_mean_squared_error')

	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	clf = clf.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])
	grid_results = clf.cv_results_
	

	#2. produce estimates
	mu_p_ts = clf.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	return mu_p_ts


def estimate_SVR(x_cal, y_cal, test_split, lreturns, dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from sklearn.svm import SVR
	from sklearn.model_selection import GridSearchCV
	import numpy as np
	from evaluation import plot_pred_true_b, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)


	#benchmark estimates -> this one is corrupt!
	bench_y = benchm_f(lreturns,dates,n_past,len(y_test))

	c_range = np.geomspace(0.1,100, 12)
	epsilon_range = np.geomspace(1e-2,12,10)
	RR_parameters = [{'C': c_range, 'epsilon': epsilon_range}]

	RR_model = SVR()
	clf = GridSearchCV(RR_model, RR_parameters,scoring='neg_mean_squared_error')

	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	clf = clf.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])
	
	

	#2. produce estimates
	mu_p_ts = clf.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	return mu_p_ts



def estimate_xgboost(x_cal, y_cal, test_split, lreturns, dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	import xgboost as xgb
	from sklearn.model_selection import GridSearchCV
	import numpy as np
	from evaluation import plot_pred_true_b, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)


	#benchmark estimates -> this one is corrupt!
	bench_y = benchm_f(lreturns,dates,n_past,len(y_test))

	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	xgb_model = xgb.XGBRegressor()
	max_depth_range = np.array(np.linspace(5,100,5),dtype=int)
	n_est_range = np.array(np.linspace(50,500,5),dtype=int)
	clf = GridSearchCV(xgb_model,{'max_depth': max_depth_range,'n_estimators': n_est_range}, verbose=1)
	clf = clf.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])
	

	#2. produce estimates
	mu_p_ts = clf.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	return mu_p_ts



def estimate_keras(x_cal, y_cal, test_split, lreturns, dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers.embeddings import Embedding
	import numpy as np
	from evaluation import plot_pred_true_b, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass


	ind_mask = np.invert(np.isnan(y_cal))
	ind_mask = np.reshape(ind_mask,[len(y_cal),1])

	x_train, y_train, x_test, y_test = train_test_split(x_cal[ind_mask[:,0],:], y_cal[ind_mask[:,0]], test_split)


	#benchmark estimates -> this one is corrupt!
	bench_y = benchm_f(lreturns,dates,n_past,len(y_test))

	


	model = Sequential()

	model.add(LSTM(64, input_shape=(x_train.shape[1],1), return_sequences=True))
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x_train[:,:,None], y_train,validation_data=(x_test[:,:,None], y_test), epochs=5, batch_size=64)


	mu_p_ts = model.predict(x_test[:,:,None], batch_size=32, verbose=0)


	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	return mu_p_ts[:,0]
