
def ind_closest(lval,list_vals):
	import numpy as np
	try:
		indt = list(list_vals).index(lval)
	except:
		ind_temp = min(list_vals, key=lambda x: abs(x - lval))
		indt = list(list_vals).index(ind_temp)
	return indt

def data_label_method_val(lreturns, cur_d,dates_stocks):
	import numpy as np
	return lreturns[ind_closest(cur_d, dates_stocks)+1,:]

#now I am taking past and future to correctly estimate var/cov
def data_label_method_cov(lreturns, cur_d,dates_stocks):
	#covariance future/past... important!!
	import numpy as np

	#important
	n = 4

	indt = ind_closest(cur_d, dates_stocks)
	try:
		ret_val = np.cov(lreturns[0,indt-n:indt+n],lreturns[1,indt-n:indt+n])[0,1]
	except:
		ret_val = np.cov(lreturns[0,indt-n:],lreturns[1,indt-n:])[0,1]
	return ret_val


def create_documents(news,lreturns,dates_stocks,ylm):
	import datetime
	import numpy as np
	from gensim.models.doc2vec import TaggedDocument

	#variables
	documents = []
	y = []
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

	return documents, np.array(y), np.array(x_dates)




def gen_xy_daily(documents,features,window,mcount,dm_opt, tables=False, dmm=0, dmc=0):
	import datetime
	import numpy as np
	from gensim.models.doc2vec import Doc2Vec
	from evaluation import doc2vec_tables

	x = []
	model = Doc2Vec(dm = dm_opt, size=features, window=window, min_count=mcount, workers=4, dbow_words=1,dm_mean=dmm,dm_concat=dmc)

	model.build_vocab(documents)
	model.train(documents,total_examples=model.corpus_count, epochs=3)

	for i in documents:
		x.append(model.infer_vector(i[0]))

	#evaluation
	if tables:
		doc2vec_tables(model, documents)

	return np.array(x)



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

def bench_mark_mu(lreturns,dates_stocks,n_past,len_o):
	import numpy as np
	ret_val = []
	for i in len_o:
		indt = ind_closest(i, dates_stocks)
		ret_val.append(np.mean(lreturns[indt-n_past:indt,:],axis=0))

	ret_val = np.array(ret_val).ravel()
	return ret_val

def bench_mark_cov(lreturns,dates_stocks,n_past,len_o):
	import numpy as np
	ret_val = []
	for i in len_o:
		indt = ind_closest(i, dates_stocks)
		ret_val.append(np.cov(lreturns[0,indt-n_past:indt],lreturns[1,indt-n_past:indt])[0,1])

	ret_val = np.array(ret_val).ravel()
	return ret_val

def calibrate_doc2vec(lreturns,dates,test_split,news_data):
	from sklearn import svm
	import numpy as np
	import gc

	#default values
	ws_def = 8
	mc_def = 10
	dm_def = 1
	dmm_def=0 
	dmc_def=0

	#parameter space for gensim 
	fts_space = np.linspace(150,900,5,dtype=int)
	ws_space = np.linspace(9,25,4,dtype=int)
	mc_space = np.linspace(0,27,4,dtype=int)
	x_dm_space = [0,1]
	x_dmm_space = [0,1]
	x_dcm_space = [0,1]

	[documents, y,dates_ret] = create_documents(news_data,lreturns,dates,data_label_method_val)

	loss = []
	x_val = []
	for fts in fts_space:
		x = gen_xy_daily(documents,fts,ws_def,mc_def, dm_def, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
		print('doc2vec model built')
	fts_opt = fts_space[np.argmax(loss)]
	del loss, x_val 
	gc.collect()

	loss = []
	x_val = []
	for ws in ws_space:
		x = gen_xy_daily(documents,fts_opt,ws,mc_def, dm_def, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
		print('doc2vec model built')
	ws_opt = ws_space[np.argmax(loss)]
	del loss, x_val 
	gc.collect()

	loss = []
	x_val = []
	for mc in mc_space:
		x = gen_xy_daily(documents,fts_opt,ws_opt,mc, dm_def, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
		print('doc2vec model built')
	mc_opt = mc_space[np.argmax(loss)]
	del loss, x_val 
	gc.collect()

	loss = []
	x_val = []
	for dm in x_dm_space:
		x = gen_xy_daily(documents,fts_opt,ws_opt,mc_opt, dm, False, dmm_def, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
		print('doc2vec model built')
	dm_opt = x_dm_space[np.argmax(loss)]
	del loss, x_val 
	gc.collect()

	loss = []
	x_val = []
	for dmm in x_dmm_space:
		x = gen_xy_daily(documents,fts_opt,ws_opt,mc_opt, dm_opt, False, dmm, dmc_def)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
		print('doc2vec model built')
	dmm_opt = x_dmm_space[np.argmax(loss)]
	del loss, x_val 
	gc.collect()

	loss = []
	x_val = []
	for dmc in x_dcm_space:
		x = gen_xy_daily(documents,fts_opt,ws_opt,mc_opt, dm_opt, False, dmm_opt, dmc)
		loss.append(val_cv_eval(x,y,test_split))
		x_val.append(x)
		print('doc2vec model built')
	dmc_opt = x_dcm_space[np.argmax(loss)]

	parameters = 'features ' + str(fts_opt) + ', window ' + str(ws_opt) + ', min. count ' + str(mc_opt) + (', PV-DM, ' if dm_opt else ', PV-DBOW, ') + ('mean, ' if dm_opt else 'sum') + ('concatenation' if dm_opt else '')
	return [np.matrix(x_val[np.argmax(loss)]),parameters], dates_ret


		



def sort_predictability(news_data,lreturns,dates,test_split,names):
	import datetime
	import numpy as np
	from sklearn.linear_model import LinearRegression
	from sklearn.model_selection import cross_val_score
	from evaluation import make_pred_sort_table

	#hard coded values... review -> definitly different
	[documents, y,_] = create_documents(news_data,lreturns,dates, data_label_method_val)
	x = gen_xy_daily(documents,340,8,21,1)

	loss_ar_svm = []

	for i in range(np.shape(y)[1]):
		clf = LinearRegression(n_jobs=number_jobs)
		ind_mask = np.invert(np.isnan(y[:,i]))

		if np.sum(ind_mask) > 0:
			scores = cross_val_score(clf, x[ind_mask,:], y[ind_mask,i], cv=5, scoring='neg_mean_squared_error',n_jobs=number_jobs)
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
		y.append(returns[ind_closest(i, dates_prices)+1])

	return np.array(y).ravel()

def produce_y_cov(returns,dates_prices,dates_news):
	import numpy as np
	import datetime

	n = 3

	dates_prices = dates_prices.astype(datetime.date)
	dates_news = list(map(datetime.datetime.date, dates_news.astype(datetime.date)))

	y = []
	for i in dates_news:
		# try:
		# 	indt = list(dates_prices).index(i)
		# except:
		# 	ind_temp = min(dates_prices, key=lambda x: abs(x - i))
		# 	indt = list(dates_prices).index(ind_temp)
		indt = ind_closest(i, dates_prices)
		try:
			ret_val = np.cov(returns[0,indt-n:indt+n],returns[1,indt-n:indt+n])[0,1]
		except:
			ret_val = np.cov(returns[0,indt-n:],returns[1,indt-n:])[0,1]

		y.append(ret_val)

	return np.array(y).ravel()

def produce_mu_cov(x, test_split, lreturns, dates_prices, dates_news, n_past, names, firm_ind_u,reg_method):
	import numpy as np
	import datetime

	show_p = False
	stables = False
	losses = 0
	r2_mat = []
	# get improved mu estimates
	mu_p_ts = np.empty((int(np.ceil(np.shape(x)[0]*test_split)),0), float)
	for i in firm_ind_u:
		temp1 = np.transpose(np.matrix( lreturns[:,i]))
		y = produce_y_ret(temp1,dates_prices, dates_news)

		if i == firm_ind_u[0]:
			stables = True
			[mu_p_ts_temp, lossest, r2t, parameters_reg] = reg_method(x, y, test_split, temp1, dates_prices, dates_news, n_past,i,bench_mark_mu, "Mean",names[i],show_p,stables)
		else:
			[mu_p_ts_temp, lossest, r2t, _] = reg_method(x, y, test_split, temp1, dates_prices, dates_news, n_past,i,bench_mark_mu, "Mean",names[i],show_p,stables)
		mu_p_ts = np.concatenate([mu_p_ts,np.reshape(mu_p_ts_temp,(np.shape(mu_p_ts)[0],1))],axis=1)
		r2_mat.append(r2t)
		losses = losses + lossest
		print(str(datetime.datetime.now())+': Successfully produced mu_p_ts for '+names[i])
		if i == firm_ind_u[0]:
			stables = False


	# get improved cov estimates
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
			[cov_p_ts[:,i,j], lossest, r2t, _] = reg_method(x, y, test_split, temp1, dates_prices, dates_news, n_past,i,bench_mark_cov, label_text, l2_test ,show_p,stables)
			cov_p_ts[:,j,i] = cov_p_ts[:,i,j]
			losses = losses + lossest
			r2_mat.append(r2t)
			print(str(datetime.datetime.now())+': Successfully produced co_p_ts for '+names[firm_ind_u[i]]+' and '+names[firm_ind_u[j]])
			if (i == j) and (i == 0):
				stables = False
			if (i == 0) and (j == 1):
				stables = False

	return mu_p_ts, cov_p_ts, losses, np.nanmean(r2_mat), parameters_reg


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
				corpus.append(str(temp_word))
				temp_word = ""
			x_dates.append(cur_d)
			prev_d = cur_d
		for hj in i[1]:
			temp_word = temp_word + hj + " "
			#words.append(hj)
	corpus.append(str(temp_word))

	
	fts_space = np.linspace(2000,7000,10,dtype=int)

	lossC = []
	lossT = []
	x_C = []
	x_T = []
	parameter1 = []
	parameter2 = []
	for i in fts_space:
		vectorizerCount = CountVectorizer(analyzer='word',ngram_range=(1,n_gram),stop_words='english',min_df=3,max_features=i)
		x = vectorizerCount.fit_transform(corpus).toarray()
		y = np.array([data_label_method_val(lreturns, cur_d, dates_stocks) for cur_d in x_dates ])
		lossC.append(val_cv_eval(x,y,test_split))
		x_C.append(x)
		parameter1.append('BOW: stopwords, cut-off=3, features='+str(i)+', '+str(n_gram)+'-gram')

		vectorizerTFIDF = TfidfVectorizer(analyzer='word',ngram_range=(1,n_gram),stop_words='english',min_df=3,max_features=i)
		x = vectorizerTFIDF.fit_transform(corpus).toarray()
		y = np.array([data_label_method_val(lreturns, cur_d,dates_stocks) for cur_d in x_dates ])
		lossT.append(val_cv_eval(x,y,test_split))
		x_T.append(x)
		parameter2.append('TFIDF: stopwords, cut-off=3, features='+str(i)+', '+str(n_gram)+'-gram')

	return [x_C[np.argmax(lossC)], parameter1[np.argmax(lossC)]], [x_T[np.argmax(lossT)], parameter1[np.argmax(lossC)]], np.array(x_dates)

def val_cv_eval(x,y,split):
	from sklearn.model_selection import cross_val_score
	from sklearn.linear_model import LinearRegression 
	import numpy as np

	ind_mask = np.invert(np.isnan(y)).ravel()

	clf = LinearRegression(n_jobs=number_jobs)
	if np.sum(ind_mask) > 0:
		#different score neg_mean_squared_error / r2 
		scores = cross_val_score(clf, x[ind_mask,:], y[ind_mask], cv=5, scoring='neg_mean_squared_error',n_jobs=number_jobs)
		return scores.mean()
	else:
		return 0

def estimate_linear(x_cal, y_cal, test_split, lreturns, dates, x_dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import r2_score
	import numpy as np
	from evaluation import plot_pred_true_b#, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)


	#benchmark estimates -> this one is corrupt!
	bench_y = benchm_f(lreturns,dates,n_past, x_dates[np.shape(x_dates)[0]-np.shape(y_test)[0]:np.shape(x_dates)[0]])

	#1. model selection with cross validation and grid search
	model = LinearRegression(n_jobs=number_jobs)
	
	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	model = model.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])
	

	#2. produce estimates
	mu_p_ts = model.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	losses = mean_squared_error(y_test, mu_p_ts)
	r2 = r2_score(y_test, mu_p_ts)

	#if tables:
	#	plot_pred_true_b(y_test,mu_p_ts,bench_y,mu_var,t_text)
		#learning_curve_plots(grid_results,clf, x_cal, y_cal,n_cpu, alpha_range,gamma_range,show_p)

	parameters = 'linear Regression'
	return mu_p_ts, losses, r2, parameters

def estimate_ridge(x_cal, y_cal, test_split, lreturns, dates, x_dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from sklearn.kernel_ridge import KernelRidge
	from sklearn.model_selection import GridSearchCV
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import r2_score
	import numpy as np
	from evaluation import plot_pred_true_b#, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)

	bench_y = benchm_f(lreturns,dates,n_past, x_dates[np.shape(x_dates)[0]-np.shape(y_test)[0]:np.shape(x_dates)[0]])

	#1. model selection with cross validation and grid search
	#alpha_range1 = np.geomspace(0.1,80, 12)
	alpha_range1 = [0.1       ,   0.18361885,   0.33715883,   0.61908719,
					1.13676079,   2.08730714,   3.83268943,   7.0375404 ,
					12.922251  ,  23.72768914,  43.56851077,  80.]
	#alpha_range = np.geomspace(1e-8,40, 12)
	#gamma_range = np.geomspace(1e-2,12,10)
	gamma_range = [ 1.00000000e-02,   2.19852420e-02,   4.83350864e-02,
					1.06265857e-01,   2.33628058e-01,   5.13636937e-01,
					1.12924323e+00,   2.48266857e+00,   5.45820693e+00,
					1.20000000e+01]
	#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise -> kernels
	RR_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'alpha': alpha_range1},
					{'kernel': ['linear'], 'alpha': alpha_range1}]

	RR_model = KernelRidge(alpha=30)
	clf = GridSearchCV(RR_model, RR_parameters,scoring='neg_mean_squared_error',n_jobs=number_jobs)

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

	losses = mean_squared_error(y_test, mu_p_ts)
	r2 = r2_score(y_test, mu_p_ts)

	if tables:
		plot_pred_true_b(grid_results,clf, x_cal, y_cal,4, alpha_range1,gamma_range,y_test,mu_p_ts,bench_y,mu_var,t_text)
		#learning_curve_plots(grid_results,clf, x_cal, y_cal,4, alpha_range1,gamma_range,show_plots)


	para_string = 'Ridge Regression:'
	for key, value in clf.best_params_.items():
		if type(value) == np.float64:
			para_string = para_string + ' ' + key + '=' + "{:.4f}".format(value) + ','
		elif type(value) == float:
			para_string = para_string + ' ' + key + '=' + "{:.4f}".format(value) + ','
		elif type(value) == int:
			para_string = para_string + ' ' + key + '=' + str(value) + ','
		elif type(value) == str:
			para_string = para_string + ' ' + key + '=' + value + ','
	return mu_p_ts, losses, r2, para_string[:-1]

def estimate_SVR(x_cal, y_cal, test_split, lreturns, dates, x_dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from sklearn.svm import SVR
	from sklearn.model_selection import GridSearchCV
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import r2_score
	import numpy as np
	from evaluation import plot_pred_true_b#, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)

	bench_y = benchm_f(lreturns,dates,n_past, x_dates[np.shape(x_dates)[0]-np.shape(y_test)[0]:np.shape(x_dates)[0]])

	#c_range = np.geomspace(0.1,100, 12)
	c_range = [	0.1       ,    0.18738174,    0.35111917,    0.65793322,
				1.23284674,    2.3101297 ,    4.32876128,    8.11130831,
				15.19911083,   28.48035868,   53.36699231,  100.        ]
	#epsilon_range = np.geomspace(1e-2,12,10)
	epsilon_range = [	1.00000000e-02,   2.19852420e-02,   4.83350864e-02,
						1.06265857e-01,   2.33628058e-01,   5.13636937e-01,
						1.12924323e+00,   2.48266857e+00,   5.45820693e+00,
						1.20000000e+01]
	RR_parameters = [{'C': c_range, 'epsilon': epsilon_range}]

	RR_model = SVR()
	clf = GridSearchCV(RR_model, RR_parameters,scoring='neg_mean_squared_error',n_jobs=number_jobs)

	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	clf = clf.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])

	#2. produce estimates
	mu_p_ts = clf.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	losses = mean_squared_error(y_test, mu_p_ts)
	r2 = r2_score(y_test, mu_p_ts)


	para_string = 'Support Vector Regression:'
	for key, value in clf.best_params_.items():
		if type(value) == np.float64:
			para_string = para_string + ' ' + key + '=' + "{:.4f}".format(value) + ','
		elif type(value) == float:
			para_string = para_string + ' ' + key + '=' + "{:.4f}".format(value) + ','
		elif type(value) == int:
			para_string = para_string + ' ' + key + '=' + str(value) + ','
		elif type(value) == str:
			para_string = para_string + ' ' + key + '=' + value + ','
	return mu_p_ts, losses, r2, para_string[:-1]

def estimate_xgboost(x_cal, y_cal, test_split, lreturns, dates, x_dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	import xgboost as xgb
	from sklearn.model_selection import GridSearchCV
	import numpy as np
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import r2_score
	from evaluation import plot_pred_true_b#, learning_curve_plots
	import datetime

	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)

	bench_y = benchm_f(lreturns,dates,n_past, x_dates[np.shape(x_dates)[0]-np.shape(y_test)[0]:np.shape(x_dates)[0]])

	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	xgb_model = xgb.XGBRegressor()
	max_depth_range = np.array(np.linspace(5,100,5),dtype=int)
	n_est_range = np.array(np.linspace(50,500,5),dtype=int)
	clf = GridSearchCV(xgb_model,{'max_depth': max_depth_range,'n_estimators': n_est_range},n_jobs=number_jobs)
	clf = clf.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])
	

	#2. produce estimates
	mu_p_ts = clf.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	losses = mean_squared_error(y_test, mu_p_ts)
	r2 = r2_score(y_test, mu_p_ts)

	para_string = 'XGBoost:'
	for key, value in clf.best_params_.items():
		if type(value) == np.float64:
			para_string = para_string + ' ' + key + '=' + "{:.4f}".format(value) + ','
		elif type(value) == float:
			para_string = para_string + ' ' + key + '=' + "{:.4f}".format(value) + ','
		elif type(value) == int:
			para_string = para_string + ' ' + key + '=' + str(value) + ','
		elif type(value) == str:
			para_string = para_string + ' ' + key + '=' + value + ','
	return mu_p_ts, losses, r2, para_string[:-1]

def estimate_keras(x_cal, y_cal, test_split, lreturns, dates, x_dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers.embeddings import Embedding
	import numpy as np
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import r2_score
	from evaluation import plot_pred_true_b#, learning_curve_plots
	import datetime


	#get x and y
	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass


	ind_mask = np.invert(np.isnan(y_cal))
	ind_mask = np.reshape(ind_mask,[len(y_cal),1])

	x_train, y_train, x_test, y_test = train_test_split(x_cal[ind_mask[:,0],:], y_cal[ind_mask[:,0]], test_split)

	bench_y = benchm_f(lreturns,dates,n_past, x_dates[np.shape(x_dates)[0]-np.shape(y_test)[0]:np.shape(x_dates)[0]])

	model = Sequential()

	model.add(LSTM(64, input_shape=(x_train.shape[1],1), return_sequences=True))
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(32, return_sequences=True))
	model.add(LSTM(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x_train[:,:,None], y_train,validation_data=(x_test[:,:,None], y_test), epochs=5, batch_size=64)


	mu_p_ts = model.predict(x_test[:,:,None], batch_size=32)

	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	losses = mean_squared_error(y_test, mu_p_ts)
	r2 = r2_score(y_test, mu_p_ts)
	parameters = 'RNN with LSTM (4 Layers:64,32,32,1)'
	return mu_p_ts[:,0], losses, r2, parameters
