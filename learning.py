

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
				x_dates.append(cur_d)
				words = []
			prev_d = cur_d
			#mu/y -> what do I want, the mean next day, average next three days
			y.append(ylm(lreturns, cur_d,dates_stocks))

		#x and skip last sentence / headlines... no last sentence only one
		for hj in i[1]:
			words.append(hj)
	
	try:
		documents.append(TaggedDocument(words,tags=str(doc_c)))
		x_dates.append(cur_d)
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


#modify the standard values... according to something
def produce_doc2vecmodels_sign(fts_space,ws_space,mc_space,lreturns,dates,test_split,news_data):
	import datetime
	import numpy as np

	#creat different training x
	# probably has to be nested....
	x_fts = []
	#parameter calibration with SVM
	for fts in fts_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts,8,10,data_label_method_sign,1)
		x_fts.append(x)
		#x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		# y_train[y_train < 0] = 0
		# y_test[y_test < 0] = 0
		
	x_ws = []
	for fts in ws_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,350,fts,10,data_label_method_sign,1)
		x_ws.append(x)
		

	x_mc = []
	for fts in mc_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,350,8,fts,data_label_method_sign, 1)
		x_mc.append(x)

	x_dm = []
	for xdmc in [0,1]:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,350,8,10,data_label_method_sign, 1)
		x_dm.append(x)

	x_dmm = []
	for dmm in [0,1]:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,350,8,10,data_label_method_sign, 1,dmm=dmm)
		x_dmm.append(x)

	x_dmc = []
	for dmc in [0,1]:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,350,8,10,data_label_method_sign, 1,dmc=dmc)
		x_dmc.append(x)


	#import pickle
	#pickle.dump([x_fts, x_ws, x_mc,y], open( "Data/diffx", "wb" ) )

	return x_fts, x_ws, x_mc, y, x_dm, x_dmm, x_dmc

def svm_sign_eval(x,y,split):
	from sklearn import svm
	import numpy as np

	x_train, y_train, x_test, y_test = train_test_split(x, y, split)

	clf = svm.SVC()
	ind_mask = np.invert(np.isnan(y_train)).ravel()

	if np.sum(ind_mask) > 0:
		clf.fit(x_train[ind_mask,:], y_train[ind_mask].ravel())
		res =  np.array(clf.predict(x_test)).flatten()
		return np.sum(np.invert(y_test.ravel() == res))/ np.shape(y_test)[0]
	else:
		return 1


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
		loss.append(svm_sign_eval(x,y,test_split))
		x_val.append(x)
	fts_opt = fts_space[np.argmin(loss)]

	loss = []
	x_val = []
	for ws in ws_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws,mc_def,data_label_method_sign, dm_def, False, dmm_def, dmc_def)
		loss.append(svm_sign_eval(x,y,test_split))
		x_val.append(x)
	ws_opt = ws_space[np.argmin(loss)]

	loss = []
	x_val = []
	for mc in mc_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc,data_label_method_sign, dm_def, False, dmm_def, dmc_def)
		loss.append(svm_sign_eval(x,y,test_split))
		x_val.append(x)
	mc_opt = mc_space[np.argmin(loss)]

	loss = []
	x_val = []
	for dm in x_dm_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc_opt,data_label_method_sign, dm, False, dmm_def, dmc_def)
		loss.append(svm_sign_eval(x,y,test_split))
		x_val.append(x)
	dm_opt = x_dm_space[np.argmin(loss)]

	loss = []
	x_val = []
	for dmm in x_dmm_space:
		[x,y,_] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc_opt,data_label_method_sign, dm_opt, False, dmm, dmc_def)
		loss.append(svm_sign_eval(x,y,test_split))
		x_val.append(x)
	dmm_opt = x_dmm_space[np.argmin(loss)]

	loss = []
	x_val = []
	for dmc in x_dcm_space:
		[x,y,dates] = gen_xy_daily(news_data,lreturns,dates,fts_opt,ws_opt,mc_opt,data_label_method_sign, dm_opt, False, dmm_opt, dmc)
		loss.append(svm_sign_eval(x,y,test_split))
		x_val.append(x)
	dmc_opt = x_dcm_space[np.argmin(loss)]

	return np.matrix(x_val[np.argmin(loss)]), dates


		
def make_pred_sort_table(firm_ind_u, loss, names, uss):
	import numpy as np

	f = open('Output/tables/pred_sort.tex', 'w')
	f.write('\\begin{tabular}{ r | l }\n')
	f.write('Stock Ticker & Loss \\\\ \n ')
	f.write('\hline \n')
	if uss > 10:
		uss = 10
	for i in range(uss):
		f.write(names[firm_ind_u[i]]+' & '+ "{:.4f}".format((loss[i]))+' \\\\ \n ')
	f.write('\hline \n')
	f.write('\hline \n')
	f.write('Mean & '+ "{:.4f}".format(np.nanmean(loss[:]))+' \\\\ \n ')
	f.write('\\end{tabular}')
	f.close() 


def sort_predictability(news_data,lreturns,dates,test_split,names,uss):
	import datetime
	import numpy as np
	#hard coded values... review
	[x,y,_] = gen_xy_daily(news_data,lreturns,dates,340,8,21,data_label_method_sign,1)
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
		ind_mask = np.invert(np.isnan(y_train[:,i]))

		if np.sum(ind_mask) > 0:
			clf.fit(x_train[ind_mask,:], y_train[ind_mask,i])
			#res =  np.reshape(np.array(clf.predict(x_test)),[1,np.shape(x_test)[0]])
			res =  np.array(clf.predict(x_test)).flatten()
			temptt = np.sum(np.invert(y_test[:,i] == res))/ np.shape(y_test)[0]# (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
			#print(temptt)
			loss_ar_svm.append(temptt)
		else:
			loss_ar_svm.append(1)


	npal = np.array(loss_ar_svm)
	firm_ind_u = np.argsort(npal[npal!=1])

	make_pred_sort_table(firm_ind_u, npal[np.argsort(npal[npal!=1])], names, uss )

	#optional information
	#names = np.array(names) 
	#npal[firm_ind_u[0:firms_used]]
	#names[firm_ind_u[0:firms_used]]

	return firm_ind_u

def inside_sxy(x, y, test_split, clf_v):
	import numpy as np

	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
	y_train[y_train < 0] = 0
	y_test[y_test < 0] = 0
	#clf = clf_v

	ind_mask = np.invert(np.isnan(y_train))

	clf_v.fit(x_train[ind_mask,:], y_train[ind_mask])
	#res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
	res = np.array(clf_v.predict(x_test)).flatten()
	return(np.sum(np.abs(np.subtract(y_test,res)))/np.shape(y_test)[0])

def doc2vec_table(lpara1,lop1, lpara2, lop2, lpara3, lop3,dm_w, dmm_w, dmc_w,opt_loss, fts_op, ws_op, mc_op, dm_op, dmm_op, dmc_op):
	import numpy as np
	import datetime

	#keep in mind the fixed values -> maybe modify them
	f = open('Output/tables/'+str(datetime.datetime.now())+'ddoc2vec.tex', 'w')
	f.write('\\begin{tabular}{ r r r r r r | l }\n')
	f.write('Method & Dim. feature vec. & Window & Min. count & Sum/mean & concatenation & Loss \\\\ \n ')
	f.write('\hline \n')
	for i in range(3):
		f.write('PV-DM & '+ str(lpara1[i]) +' & 8 & 10 & sum & Off & '+ "{:.4f}".format((lop1[i]))+' \\\\ \n ')
	for i in range(3):
		f.write('PV-DM & '+'350 & '+ str(lpara2[i]) +' & 10 &sum & Off & '+ "{:.4f}".format((lop2[i]))+' \\\\ \n ')
	for i in range(3):
		f.write('PV-DM & '+'350 & 8 & '+str(lpara3[i]) +' &sum & Off & '+ "{:.4f}".format((lop3[i]))+' \\\\ \n ')

	f.write('PV-DBOW & '+'350 & 8 & 10 &sum & Off & '+ "{:.4f}".format((dm_w))+' \\\\ \n ')
	f.write('PV-DM & '+'350 & 8 & 10 &mean & Off & '+ "{:.4f}".format((dmm_w))+' \\\\ \n ')
	f.write('PV-DM & '+'350 & 8 & 10 &mean & On & '+ "{:.4f}".format((dmc_w))+' \\\\ \n ')
	#final line -> best
	f.write('\hline \n')
	f.write('\hline \n')

	f.write(('PV-DM & ' if dm_op else 'PV-DBOW & ') + str(fts_op) + ' & ' + str(ws_op) + ' & ' + str(mc_op) + ' & ' + ('mean & ' if dmm_op else 'sum & ') + ('Off & ' if dmm_op else 'On & ') + "{:.4f}".format((opt_loss))+' \\\\ \n ')
	f.write('\\end{tabular}')
	f.close() 


def stock_xy(test_split, fts_space,ws_space, mc_space, news_data,lreturns,dates,x_fts, x_ws, x_mc,y,m_eval,clf_v,x_dm, x_dmm, x_dmc, tables = False):
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
	for x in x_fts + x_ws + x_mc + x_dm + x_dmm + x_dmc:
		loss_cali[0].append(inside_sxy(x, y, test_split, clf_v))

	
	fts_w, xws_w, xmc_w = np.split(np.array(loss_cali[0])[:-6],3)
	dm_w, dmm_w, dmc_w = np.split(np.array(loss_cali[0])[-6:],3)
	fts_op = fts_space[np.argmin(fts_w)]
	ws_op = ws_space[np.argmin(xws_w)]
	mc_op = mc_space[np.argmin(xmc_w)]
	dm_op = fts_space[np.argmin(dm_w)]
	dmm_op = ws_space[np.argmin(dmm_w)]
	dmc_op = mc_space[np.argmin(dmc_w)]


	[x_cal,y_cal,x_dates] = gen_xy_daily(news_data,lreturns,dates,fts_op,ws_op,mc_op,m_eval,dm_op,tables, dmm=dmm_op, dmc=dmc_op)

	opt_loss = inside_sxy(x_cal, y, test_split, clf_v)
	if tables:
		doc2vec_table(fts_space[np.argsort(fts_w)[:3]], np.sort(fts_w)[:3],
			ws_space[np.argsort(xws_w)[:3]], np.sort(xws_w)[:3],
			mc_space[np.argsort(xmc_w)[:3]], np.sort(xmc_w)[:3], 
			dm_w[1], dmm_w[1], dmc_w[1],opt_loss, fts_op, ws_op, mc_op, dm_op, dmm_op, dmc_op)

	

	return x_cal,y_cal,x_dates




def mu_news_estimate(x_cal, y_cal, test_split, lreturns, dates, n_past, ind_r,benchm_f,mu_var,t_text,show_plots,tables):
	from sklearn.linear_model import Ridge
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

	#remove nan
	#ind_mask = np.invert(np.isnan(y_train[i]))

	ind_mask = np.invert(np.isnan(y_train))
	ind_mask = np.reshape(ind_mask,[len(y_train),1])

	clf = clf.fit(x_train[ind_mask[:,0],:], y_train[ind_mask[:,0]])
	grid_results = clf.cv_results_
	

	#2. produce estimates
	#clf = clf.fit(x_train, y_train)
	mu_p_ts = clf.predict(x_test)
	#set variance to zero if negativ / probably a bad trick
	if benchm_f == bench_mark_cov:
		if  np.array_equal(lreturns[0,:], lreturns[1,:]):
			mu_p_ts[mu_p_ts < 0] = 0

	#3. plots
	#plot_pred_true_b(y_test,clf.predict(x_test),bench_y.flatten(), mu_var, t_text) 
	#learning_plots(grid_results,clf, x_cal, y_cal,1,alpha_range,gamma_range,show_plots)

	# if tables:
	# 	learning_curve_plots(grid_results,clf, x_cal, y_cal,1,alpha_range1,gamma_range,show_plots)
	# 	ind_m = np.argsort(np.abs(clf.cv_results_['mean_test_score']))[1:10]

	# 	f = open('Output/tables/'+str(datetime.datetime.now())+'grdisearch.tex', 'w')
	# 	f.write('\\begin{tabular}{ r r r | l }\n')
	# 	f.write('Gamma & Alpha & Kernel & Loss \\\\ \n ')
	# 	f.write('\hline \n')
	# 	for i in ind_m:
	# 		loss_v = "{:.4f}".format((np.abs(clf.cv_results_['mean_test_score'][i])))
	# 		#gamma
	# 		try:
	# 			gamma_v = "{:.4f}".format((clf.cv_results_['params'][i]['gamma']))
	# 		except:
	# 			gamma_v = '-'
	# 		#alpha
	# 		try:
	# 			alpha_v = "{:.4f}".format((clf.cv_results_['params'][i]['alpha']))
	# 		except:
	# 			alpha_v = '-'
	# 		#kernel
	# 		try:
	# 			kernel_v = clf.cv_results_['params'][i]['kernel']
	# 		except: 
	# 			kernel_v = '-'
	# 		f.write(gamma_v + ' & ' + alpha_v + ' & '+kernel_v+' & '+loss_v+'\\\\ \n ')
	# 	f.write('\hline \n')
	# 	f.write('\hline \n')
	# 	try:
	# 		gamma_v = "{:.4f}".format((clf.best_params_['gamma']))
	# 	except:
	# 		gamma_v = '-'
	# 	#alpha
	# 	try:
	# 		alpha_v = "{:.4f}".format((clf.best_params_['alpha']))
	# 	except:
	# 		alpha_v = '-'
	# 	#kernel
	# 	try:
	# 		kernel_v = clf.best_params_['kernel']
	# 	except: 
	# 		kernel_v = '-'
	# 	f.write(gamma_v + ' & ' + alpha_v + ' & '+kernel_v+' & '+"{:.4f}".format((np.abs(clf.cv_results_['mean_test_score'][np.argsort(np.abs(clf.cv_results_['mean_test_score']))[0]])))+'\\\\ \n ')
	# 	f.write('\\end{tabular}')
	# 	f.close() 

	return mu_p_ts

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



def produce_mu_cov(x, test_split, lreturns, dates_prices, dates_news, n_past, names, firm_ind_u):
	import numpy as np

	show_p = False
	stables = False
	# 5. single stock parameter calibration & get improved mu estimates
	mu_p_ts = np.empty((int(np.ceil(np.shape(x)[0]*test_split)),0), float)
	for i in firm_ind_u:
		if i == firm_ind_u[0]:
			stables = True
		temp1 = np.transpose(np.matrix( lreturns[:,i]))
		#[x_cal, y_cal, x_dates] = stock_xy(test_split,fts_space,ws_space, mc_space,news_data,temp1,dates,x_fts, x_ws, x_mc,y[:,i],data_label_method_val,svm.SVC(),x_dm, x_dmm, x_dmc,tables=stables)
		y = produce_y_ret(temp1,dates_prices, dates_news)

		mu_p_ts = np.concatenate([mu_p_ts,np.reshape(mu_news_estimate(x_calibrated, y, test_split, temp1, dates_prices, n_past,i,bench_mark_mu, "Mean",names[i],show_p,stables),(np.shape(mu_p_ts)[0],1))],axis=1)
		del y, temp1
		gc.collect()
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
			#[_,y,_] = gen_xy_daily(news_data,temp1,dates,220,8,10,data_label_method_cov,1) 
			y = produce_y_cov(temp1,dates_prices, dates_news)
			#[x_cal, y_cal, x_dates] = stock_xy(test_split,fts_space,ws_space, mc_space,news_data,temp1,dates,x_fts, x_ws, x_mc,y,data_label_method_cov,Ridge(alpha=0),x_dm, x_dmm, x_dmc, tables= stables)
			if i == j:
				label_text = "Variance"
				l2_test = names[i]
			else:
				label_text = "Covariance"
				l2_test = names[i] + " and " + names[j]
			cov_p_ts[:,i,j] = mu_news_estimate(x_calibrated, y, test_split, temp1, dates_prices, n_past,i,bench_mark_cov, label_text, l2_test ,show_p,stables)
			cov_p_ts[:,j,i] = cov_p_ts[:,i,j]
			#del x_cal, y_cal, temp1, y
			gc.collect()
			print(str(datetime.datetime.now())+': Successfully produced co_p_ts for '+names[firm_ind_u[i]]+' and '+names[firm_ind_u[j]])
			if (i == j) and (i == 0):
				stables = False
			if (i == 0) and (j == 1):
				stables = False
	return mu_p_ts, cov_p_ts


#create corpus
def tfidf_vector(n_gram, news_data):
	import numpy as np
	from sklearn.feature_extraction.text import TfidfVectorizer
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

	
	vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,n_gram),stop_words='english',min_df=3,max_features=800)
	return vectorizer.fit_transform(corpus).toarray(), x_dates

