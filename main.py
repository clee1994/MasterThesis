
from evaluation import *
from learning import *

from sklearn import svm
from sklearn.linear_model import Ridge

import datetime, pickle, gc, random
import numpy as np


# 0. modifiable variables
firms_used = 2
n_past = 80

#traning splits
test_split = 0.25

#doc2vec spaces
fts_space = np.linspace(150,650,4,dtype=int)
ws_space = np.linspace(2,25,4,dtype=int)
mc_space = np.linspace(0,50,4,dtype=int)

#fts_space = np.linspace(150,650,8,dtype=int)
#ws_space = np.linspace(2,25,8,dtype=int)
#mc_space = np.linspace(0,50,8,dtype=int)



# 1. load and preprocess data
print(str(datetime.datetime.now())+': Start reading in news:')
news_data = pickle.load(open( "Data/Reuters.p", "rb" ) )
gc.collect()
print(str(datetime.datetime.now())+': Successfully read all news')

print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[_, dates, names, lreturns] = pickle.load(open( "Data/SP500.p", "rb" ) )
gc.collect()
print(str(datetime.datetime.now())+': Successfully read all data')


# 3. produce possible doc2vec models
[x_fts, x_ws, x_mc, y, x_dm, x_dmm, x_dmc] = produce_doc2vecmodels_sign(fts_space,ws_space,mc_space,lreturns,dates,test_split,news_data)
gc.collect()
print(str(datetime.datetime.now())+': Successfully produce doc2vec model sign')


# 4. select stocks

#cherry picking
firm_ind_u = sort_predictability(news_data,lreturns,dates,test_split,names,firms_used)[0:firms_used]
print(str(datetime.datetime.now())+': Successfully sorted')

#random
#firm_ind_u = random.sample(range(len(names)-1), firms_used)


show_p = False
stables = False
# 5. single stock parameter calibration & get improved mu estimates
mu_p_ts = np.empty((int(np.ceil(np.shape(y)[0]*test_split)),0), float)
for i in firm_ind_u:
	if i == firm_ind_u[0]:
		stables = True
	temp1 = np.transpose(np.matrix( lreturns[:,i]))
	[x_cal, y_cal, x_dates] = stock_xy(test_split,fts_space,ws_space, mc_space,news_data,temp1,dates,x_fts, x_ws, x_mc,y[:,i],data_label_method_val,svm.SVC(),x_dm, x_dmm, x_dmc,tables=stables)
	mu_p_ts = np.concatenate([mu_p_ts,mu_news_estimate(x_cal, y_cal, test_split, temp1, dates, n_past,i,bench_mark_mu, "Mean",names[i],show_p,stables)],axis=1)
	del x_cal, y_cal, x_dates, temp1
	gc.collect()
	print(str(datetime.datetime.now())+': Successfully produced mu_p_ts for '+names[i])
	if i == firm_ind_u[0]:
		stables = False


# 7. single stock parameter calibration & get improved cov estimates
cov_p_ts = np.zeros([int(np.ceil(np.shape(y)[0]*test_split)),len(firm_ind_u),len(firm_ind_u)])
for i in range(len(firm_ind_u)):
	for j in range(i+1):
		if (i == j) and (i == 0):
			stables = True
		if (i == 0) and (j == 1):
			stables = True
		temp1 = np.transpose(np.matrix( lreturns[:,[i,j]]))
		[_,y,_] = gen_xy_daily(news_data,temp1,dates,220,8,10,data_label_method_cov,1) 
		[x_cal, y_cal, x_dates] = stock_xy(test_split,fts_space,ws_space, mc_space,news_data,temp1,dates,x_fts, x_ws, x_mc,y,data_label_method_cov,Ridge(alpha=0),x_dm, x_dmm, x_dmc, tables= stables)
		if i == j:
			label_text = "Variance"
			l2_test = names[i]
		else:
			label_text = "Covariance"
			l2_test = names[i] + " and " + names[j]
		cov_p_ts[:,i,j] = mu_news_estimate(x_cal, y_cal, test_split, temp1, dates, n_past,i,bench_mark_cov, label_text, l2_test ,show_p,stables)
		cov_p_ts[:,j,i] = cov_p_ts[:,i,j]
		del x_cal, y_cal, temp1, y
		gc.collect()
		print(str(datetime.datetime.now())+': Successfully produced co_p_ts for '+names[firm_ind_u[i]]+' and '+names[firm_ind_u[j]])
		if (i == j) and (i == 0):
			stables = False
		if (i == 0) and (j == 1):
			stables = False

del news_data, x_fts, x_ws, x_mc
gc.collect()



# 6. standard past observation past mu and cov
split_point = int(np.floor(np.shape(x_dates)[0]*(1-test_split)))
pmu_p_ts = mu_gen_past1(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], n_past)
pcov_p_ts = cov_gen_past(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], n_past)
gc.collect()


# 7. build portfolios based on both
[r1,first_line] = evaluate_portfolio(names[firm_ind_u],x_dates[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates,None, None, -1)
[r2,second_line] = evaluate_portfolio(names[firm_ind_u],x_dates[(split_point+1):],lreturns,mu_p_ts,cov_p_ts,firm_ind_u,dates,None, None, -1)
[r3,third_line] = evaluate_portfolio(names[firm_ind_u],x_dates[(split_point+1):],lreturns,mu_p_ts,cov_p_ts,firm_ind_u,dates,None, 0.5, -1)
[r4,sp500] = pure_SP(x_dates[(split_point+1):])

del dates, names, lreturns, firm_ind_u, x_dates, mu_p_ts, pmu_p_ts, pcov_p_ts, split_point
gc.collect()


# 8. plotting the final results
final_plots([first_line,second_line,third_line,sp500],[r'min. var. portfolio (past obs.)', r'min. var. portfolio (doc2vec)',r'min. var. (doc2vec, l1)',r'SP500 raw performance'])

#return plots
final_plots_s([r1,r2,r3,r4],[r'past obs.', r'doc2vec',r'doc2vec, l1',r'SP500'])


#portfolio metrics tabel
# mean / variance / alpha / beta / VaR95 / sharpe ratio / Treynor / Jensen / 

#assuming some risk free rate


