from data_loading import load_news_data, load_SP_data 
from learning import gen_xy_daily, train_test_split
from evaluation import plot_pred_true, evaluate_portfolio
from stocks.stocks_big import stocks_used
import numpy as np
import datetime 
import pickle


#modifiable variables
path_to_news_files = "./Data_small/ReutersNews106521"
firms_used = 25

#traning splits
test_split = 0.15

#doc2vec spaces
fts_space = np.linspace(180,440,8,dtype=int)
ws_space = np.linspace(4,18,8,dtype=int)
mc_space = np.linspace(0,35,8,dtype=int)

#load and preprocess data
print(str(datetime.datetime.now())+': Start reading in news:')
[news_data, faulty_news] = load_news_data(path_to_news_files,False)
print(str(datetime.datetime.now())+': Successfully read all news')


print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[prices, dates, names, lreturns] = load_SP_data(stocks_used)
print(str(datetime.datetime.now())+': Successfully read all data')


#pickle dump and load
#import pickle
#pickle.dump([news_data, prices, dates, names, lreturns], open( "small_raw", "wb" ) )

#[news_data, prices, dates, names, lreturns] = pickle.load( open( "small_raw", "rb" ) )


#ht: 2 headline, 8 text
#350,6,10 -> server results... no idea why
print(str(datetime.datetime.now())+': Start generating xy:')
[x,y] = gen_xy_daily(news_data,lreturns,dates,340,8,21,2)
x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
print(str(datetime.datetime.now())+': Successfully generated xy')


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
	res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
	temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
	#print(temptt)
	loss_ar_svm.append(temptt)


npal = np.array(loss_ar_svm)
firm_ind_u = np.argsort(npal)

names = np.array(names) 
npal[firm_ind_u[0:firms_used]]
names[firm_ind_u[0:firms_used]]






#fit best model for each stock


all_loss = []
sum_loss = []

#creat different training x
# probably has to be nested....
x_fts = []
#parameter calibration with SVM
for fts in fts_space:
	print(str(datetime.datetime.now())+': Start generating xy:')
	[x,y] = gen_xy_daily(news_data,lreturns,dates,fts,8,10,2)
	x_fts.append(x)
	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
	print(str(datetime.datetime.now())+': Successfully generated xy')
	y_train[y_train < 0] = 0
	y_test[y_test < 0] = 0
	


x_ws = []
for fts in ws_space:
	print(str(datetime.datetime.now())+': Start generating xy:')
	[x,y] = gen_xy_daily(news_data,lreturns,dates,350,fts,10,2)
	x_ws.append(x)
	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
	print(str(datetime.datetime.now())+': Successfully generated xy')
	y_train[y_train < 0] = 0
	y_test[y_test < 0] = 0
	

x_mc = []
for fts in mc_space:
	print(str(datetime.datetime.now())+': Start generating xy:')
	[x,y] = gen_xy_daily(news_data,lreturns,dates,350,8,fts,2)
	x_mc.append(x)
	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
	print(str(datetime.datetime.now())+': Successfully generated xy')
	y_train[y_train < 0] = 0
	y_test[y_test < 0] = 0
	

import pickle
pickle.dump([x_fts, x_ws, x_mc], open( "Data/diffx", "wb" ) )

[x_fts, x_ws, x_mc] = pickle.load( open( "Data/diffx", "rb" ) )


#single stock parameter calibration

loss_cali = []
y_cal = []
x_cal = []
for j in range(firms_used):
	i = firm_ind_u[j]
	loss_cali.append([])
	for x in x_fts:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = svm.SVC()
		clf.fit(x_train, y_train[:,i])
		res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
		loss_cali[j].append(temptt)

	for x in x_ws:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = svm.SVC()
		clf.fit(x_train, y_train[:,i])
		res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
		loss_cali[j].append(temptt)

	for x in x_mc:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = svm.SVC()
		clf.fit(x_train, y_train[:,i])
		res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
		loss_cali[j].append(temptt)



y_cal = []
x_cal = [] 
for j in range(firms_used):
	i = firm_ind_u[j]
	#build the right x data for y
	

	fts_w, xws_w, xmc_w = np.split(np.array(loss_cali[j]),3)
	fts_op = fts_space[np.argmin(fts_w)]
	ws_op = ws_space[np.argmin(xws_w)]
	mc_op = mc_space[np.argmin(xmc_w)]
	[x,y] = gen_xy_daily(news_data,lreturns,dates,fts_op,ws_op,mc_op,2,data_label_method_val)
	y_cal.append(y[:,i])
	x_cal.append(x)


###
#lets see how much info we actually extract -> this was just checking
# for i in range(25):
# 	#i = firm_ind_u[j]
# 	x_train, y_train, x_test, y_test = train_test_split(x_cal[i], y_cal[i], test_split)
# 	y_train[y_train < 0] = 0
# 	y_test[y_test < 0] = 0
# 	clf = svm.SVC()
# 	clf.fit(x_train, y_train)
# 	res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
# 	temptt = (np.sum(np.abs(np.subtract(y_test,res)))/np.shape(y_test)[0])
# 	print(temptt)
###

#ridge regression + kernel for every stock -> calibration
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

loss_rm = []
for i in range(firms_used): 

	#3000 standard
	x_train, y_train, x_test, y_test = train_test_split(x_cal[i], y_cal[i], test_split)
	parameters = { 'alpha':[0.1,1,5,10,30]}

	modrr = Ridge(alpha=30)
	clf = GridSearchCV(modrr, parameters)
	#clf = KernelRidge(alpha=30, kernel="rbf",kernel_params =[.1,(1e-05, 100000.0)]) -> also not super useful
	#clf = linear_model.Lasso(alpha = 0) -> not really usefull

	#print(cross_val_score(clf, x_cal[i], y_cal[i], cv=5))
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	#print(mean_squared_error(y_test, y_test))


	plot_pred_true(y_test,clf.predict(x_test))
	res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
	temptt = np.mean(np.abs(np.subtract(y_test,res)))
	loss_rm.append(temptt)
	print(temptt)




sess = tf.InteractiveSession()
np.mean(mean_squared_error(y_test,clf.predict(x_test)).eval())


#actually getting estimates



#portfolio damn yeahs



