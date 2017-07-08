

#now i am taking next days return, 
#so taking todays news to evaluate what is going to happen tomorrow

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
	n = 3

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

def gen_xy_daily(news,lreturns,dates_stocks,features,window,mcount,ht,ylm):
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
	x_dates = []

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
				x_dates.append(cur_d)
				words = []
			prev_d = cur_d
			#mu/y -> what do I want, the mean next day, average next three days
			y.append(ylm(lreturns, cur_d,dates_stocks))

		#x and skip last sentence / headlines... no last sentence only one
		for j in range(len(i[ht])):
			for hj in i[ht][j]:
				 words.append(hj)
	
	try:
		documents.append(TaggedDocument(words,tags=str(cur_d)))
		x_dates.append(cur_d)
	except:
		pass

	model = Doc2Vec(size=features, window=window, min_count=mcount, workers=1)

	model.build_vocab(documents)

	model.train(documents,total_examples=model.corpus_count, epochs=model.iter)


	for i in documents:
		x.append(model.infer_vector(i[0]))

	return [np.array(x), np.array(y),np.array(x_dates)]



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



def produce_doc2vecmodels_sign(fts_space,ws_space,mc_space,lreturns,dates,test_split,news_data):
	import datetime
	import numpy as np

	#creat different training x
	# probably has to be nested....
	x_fts = []
	#parameter calibration with SVM
	for fts in fts_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts,8,10,2,data_label_method_sign)
		x_fts.append(x)
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		
	x_ws = []
	for fts in ws_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,350,fts,10,2,data_label_method_sign)
		x_ws.append(x)
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		

	x_mc = []
	for fts in mc_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,350,8,fts,2,data_label_method_sign)
		x_mc.append(x)
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0

	#import pickle
	#pickle.dump([x_fts, x_ws, x_mc,y], open( "Data/diffx", "wb" ) )

	return x_fts, x_ws, x_mc, y
		

def sort_predictability(news_data,lreturns,dates,test_split):
	import datetime
	import numpy as np
	[x,y,_] = gen_xy_daily(news_data,lreturns,dates,340,8,21,2,data_label_method_sign)
	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)


	#stock picking
	#classification
	y_train[y_train < 0] = 0
	y_test[y_test < 0] = 0

	loss_ar_svm = []

	from sklearn import svm
	for i in range(np.shape(y_train)[1]):

		#classification

		#SVM
		clf = svm.SVC()
		clf.fit(x_train, y_train[:,i])
		#res =  np.reshape(np.array(clf.predict(x_test)),[1,np.shape(x_test)[0]])
		res =  np.array(clf.predict(x_test)).flatten()
		temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
		#print(temptt)
		loss_ar_svm.append(temptt)


	npal = np.array(loss_ar_svm)
	firm_ind_u = np.argsort(npal)

	#optional information
	#names = np.array(names) 
	#npal[firm_ind_u[0:firms_used]]
	#names[firm_ind_u[0:firms_used]]

	return firm_ind_u


def stock_xy(test_split, fts_space,ws_space, mc_space, news_data,lreturns,dates,x_fts, x_ws, x_mc,y,m_eval,clf_v):
	#single stock parameter calibration
	#import pickle
	import numpy as np
	import datetime

	#[x_fts, x_ws, x_mc,y] = pickle.load( open( "Data/diffx", "rb" ) )
	loss_cali = []
	y_cal = []
	x_cal = []
	# for j in range(firms_used):
	# 	i = firm_ind_u[j]
	loss_cali.append([])

	#make it a grid search
	for x in x_fts:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = clf_v
		clf.fit(x_train, y_train)
		#res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		res = np.array(clf.predict(x_test)).flatten()
		temptt = (np.sum(np.abs(np.subtract(y_test,res)))/np.shape(y_test)[0])
		loss_cali[0].append(temptt)

	for x in x_ws:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = clf_v
		clf.fit(x_train, y_train)
		#res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		res = np.array(clf.predict(x_test)).flatten()
		temptt = (np.sum(np.abs(np.subtract(y_test,res)))/np.shape(y_test)[0])
		loss_cali[0].append(temptt)

	for x in x_mc:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = clf_v
		clf.fit(x_train, y_train)
		#res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		res = np.array(clf.predict(x_test)).flatten()
		temptt = (np.sum(np.abs(np.subtract(y_test,res)))/np.shape(y_test)[0])
		loss_cali[0].append(temptt)


	#for j in range(firms_used):
	#i = firm_ind_u[j]
	#build the right x data for y
	
	fts_w, xws_w, xmc_w = np.split(np.array(loss_cali[0]),3)
	fts_op = fts_space[np.argmin(fts_w)]
	ws_op = ws_space[np.argmin(xws_w)]
	mc_op = mc_space[np.argmin(xmc_w)]
	#print(str(fts_op)+" "+str(ws_op)+" "+str(mc_op)+" ")
	[x_cal,y_cal,x_dates] = gen_xy_daily(news_data,lreturns,dates,fts_op,ws_op,mc_op,2,m_eval)

	return x_cal,y_cal,x_dates


def mu_news_estimate(x_cal, y_cal, test_split, lreturns, dates, n_past, ind_r,benchm_f):
	#ridge regression + kernel for every stock -> calibration
	from sklearn.linear_model import Ridge
	from sklearn.kernel_ridge import KernelRidge
	from sklearn.model_selection import GridSearchCV
	import numpy as np
	from evaluation import plot_pred_true_b


	try:
		y_cal = np.reshape(y_cal, [np.shape(y_cal)[0],np.shape(y_cal)[1]])
	except:
		pass

	x_train, y_train, x_test, y_test = train_test_split(x_cal, y_cal, test_split)
	#print(str(np.shape(y_train)))

	#benchmark_might be changed
	bench_y = benchm_f(lreturns,dates,n_past,len(y_test))

	loss_rm = []
	#mu_p_ts = np.zeros((len(y_test),firms_used))
	#for i in range(firms_used): 

	#3000 standard
	#x_train, y_train, x_test, y_test = train_test_split(x_cal[0], y_cal[0], test_split)
	#parameters = { 'alpha':[0.1,0.5,1,5,10,20,30,35,40,45,50,55]}
	#parameters = { 'alpha':[0.1,0.5,1,5,10,20,30,35,40,45,50,55]}

	#param_grid={"alpha":[0.01,0.1,0.5,1,5,10,20,30,35,40,45,50,55],
				#"linear": [1e0, 0.1, 1e-2, 1e-3],
	#			"gamma": np.logspace(-2, 2, 5),
				#"rbf:": [1,3]
	#			"kernel": [(l, p)
    #                    for l in np.logspace(-2, 2, 10)
    #                   for p in np.logspace(0, 2, 10)]
	#			}

	alpha_r = [0.001,0.1,0.5,1,5,10,20,30,35,40,45,50,100,200]
	gamma_r = [1e-3, 1e-4,0.01,0.1,1,2,10]
	para2 = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'alpha': alpha_r},
			{'kernel': ['linear'], 'alpha': alpha_r}]

	modrr = KernelRidge(alpha=30)
	clf = GridSearchCV(modrr, para2)
	#clf = KernelRidge(alpha=30, kernel="rbf",kernel_params =[.1,(1e-05, 100000.0)]) -> also not super useful
	#clf = linear_model.Lasso(alpha = 0) -> not really usefull

	#print(cross_val_score(clf, x_cal[i], y_cal[i], cv=5))
	clf.fit(x_train, y_train)
	mu_p_ts = clf.predict(x_test)


	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0
	#print(mean_squared_error(y_test, y_test))

	#print(np.shape(bench_y))

	plot_pred_true_b(y_test,clf.predict(x_test),bench_y.flatten())
	#res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
	res = np.array(clf.predict(x_test)).flatten()
	temptt = np.mean(np.abs(np.subtract(y_test,res)))
	loss_rm.append(temptt)
	#print(temptt)

	return mu_p_ts




