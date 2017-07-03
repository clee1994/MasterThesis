from data_loading import load_news_data, load_SP_data 
from learning import * #gen_xy_daily, train_test_split
from evaluation import plot_pred_true, evaluate_portfolio
from stocks.stocks_big import stocks_used
import numpy as np
import datetime 
import pickle


#modifiable variables
path_to_news_files = "./Data_small/ReutersNews106521"
firms_used = 3

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
import pickle
pickle.dump([news_data, prices, dates, names, lreturns], open( "small_raw", "wb" ) )

[news_data, prices, dates, names, lreturns] = pickle.load( open( "small_raw", "rb" ) )


#ht: 2 headline, 8 text
#350,6,10 -> server results... no idea why
firm_ind_u = sort_predictability(news_data,lreturns,dates,test_split)


#fit best model for each stock
[x_fts, x_ws, x_mc, y] = produce_doc2vecmodels_sign(fts_space,ws_space,mc_space,lreturns,dates,test_split,news_data)


#single stock parameter calibration
[x_cal, y_cal, x_dates] = stock_xy(firms_used,test_split)



#ridge regression + kernel for every stock -> calibration
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


x_train, y_train, x_test, y_test = train_test_split(x_cal[0], y_cal[0], test_split)

bench_y = bench_mark_mu(lreturns,dates,70,len(y_test))

loss_rm = []
mu_p_ts = np.zeros((len(y_test),firms_used))
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
	mu_p_ts[:,i] = y_pred
	#print(mean_squared_error(y_test, y_test))


	plot_pred_true_b(y_test,clf.predict(x_test),bench_y[:,firm_ind_u[i]])
	#res =  np.reshape(np.array(clf.predict(x_test)),[1,387])
	res = np.array(clf.predict(x_test)).flatten()
	temptt = np.mean(np.abs(np.subtract(y_test,res)))
	loss_rm.append(temptt)
	#print(temptt)


#gen mu ts
split_point = int(np.floor(np.shape(x_dates)[0]*(1-test_split)))


pmu_p_ts = mu_gen_past(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], 50)


#gen cov ts
pcov_p_ts = cov_gen_past(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], 50)


#plug it into portfolio


[_,_] = evaluate_portfolio(np.array(names)[firm_ind_u[0:firms_used]],x_dates,lreturns,pmu_p_ts,pcov_p_ts)


