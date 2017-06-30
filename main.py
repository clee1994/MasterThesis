from data_loading import load_news_data, load_SP_data 
from learning import gen_xy_daily, train_test_split
from evaluation import plot_pred_true, evaluate_portfolio
from stocks_big import stocks_used
import numpy as np
import datetime 


#modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
firms_used = 505

#traning splits
test_split = 0.15


#load and preprocess data
print(str(datetime.datetime.now())+': Start reading in news:')
[news_data, faulty_news] = load_news_data(path_to_news_files,False)
print(str(datetime.datetime.now())+': Successfully read all news')


print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[prices, dates, names, lreturns] = load_SP_data(stocks_used, firms_used)
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

npal[firm_ind_u[0:25]]
names[firm_ind_u[0:25]]






#fit best model for each stock


all_loss = []
sum_loss = []

#creat different training x

x_fts = []
#parameter calibration with SVM
for fts in np.linspace(180,440,8,dtype=int):
	print(str(datetime.datetime.now())+': Start generating xy:')
	[x,y] = gen_xy_daily(news_data,lreturns,dates,fts,8,10,2)
	x_fts.append(x)
	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
	print(str(datetime.datetime.now())+': Successfully generated xy')
	y_train[y_train < 0] = 0
	y_test[y_test < 0] = 0
	


x_ws = []
for fts in np.linspace(4,12,8,dtype=int):
	print(str(datetime.datetime.now())+': Start generating xy:')
	[x,y] = gen_xy_daily(news_data,lreturns,dates,350,fts,10,2)
	x_ws.append(x)
	x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
	print(str(datetime.datetime.now())+': Successfully generated xy')
	y_train[y_train < 0] = 0
	y_test[y_test < 0] = 0
	

x_mc = []
for fts in np.linspace(0,25,8,dtype=int):
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
for i in arange(25):
	loss_cali.append([])
	for x in x_fts:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = svm.SVC()
		clf.fit(x_train, y_train[:,i])
		res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
		loss_cali[i].append(temptt)

	for x in x_ws:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = svm.SVC()
		clf.fit(x_train, y_train[:,i])
		res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
		loss_cali[i].append(temptt)

	for x in x_mc:
		x_train, y_train, x_test, y_test = train_test_split(x, y, test_split)
		y_train[y_train < 0] = 0
		y_test[y_test < 0] = 0
		clf = svm.SVC()
		clf.fit(x_train, y_train[:,i])
		res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
		temptt = (np.sum(np.abs(np.subtract(y_test[:,i],res)))/np.shape(y_test[:,i])[0])
		loss_cali[i].append(temptt)






#ridge regression + kernel for every stock -> calibration
from sklearn.linear_model import Ridge


#3000 standard
clf = Ridge(alpha=50)
clf.fit(x_train, y_train)


plot_pred_true(y_test,clf.predict(x_test))

import tensorflow as tf
from keras.losses import mean_squared_error
sess = tf.InteractiveSession()
np.mean(mean_squared_error(y_test,clf.predict(x_test)).eval())


#actually getting estimates



#portfolio damn yeahs



