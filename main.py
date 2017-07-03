from data_loading import *
from learning import * 
from evaluation import *
from stocks.stocks_big import stocks_used
import numpy as np
import datetime 
import gc


# 0. modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
firms_used = 25
n_past = 100

#traning splits
test_split = 0.15

#doc2vec spaces
#fts_space = np.linspace(180,440,8,dtype=int)
#ws_space = np.linspace(4,18,8,dtype=int)
#mc_space = np.linspace(0,35,8,dtype=int)

fts_space = np.linspace(150,650,16,dtype=int)
ws_space = np.linspace(2,22,16,dtype=int)
mc_space = np.linspace(0,50,16,dtype=int)





# 1. load and preprocess data
print(str(datetime.datetime.now())+': Start reading in news:')
[news_data, _] = load_news_data(path_to_news_files,False)
gc.collect()
print(str(datetime.datetime.now())+': Successfully read all news')

print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[_, dates, names, lreturns] = load_SP_data(stocks_used)
gc.collect()
print(str(datetime.datetime.now())+': Successfully read all data')



# 3. select stocks
firm_ind_u = sort_predictability(news_data,lreturns,dates,test_split)[0:firms_used]
print(str(datetime.datetime.now())+': Successfully sorted')


# 4. produce possible doc2vec models
[x_fts, x_ws, x_mc, y] = produce_doc2vecmodels_sign(fts_space,ws_space,mc_space,lreturns,dates,test_split,news_data)
gc.collect()
print(str(datetime.datetime.now())+': Successfully produce doc2vec model sign')



# 5. single stock parameter calibration & get improved mu estimates
mu_p_ts = np.empty((np.ceil(np.shape(y)[0]*test_split),0), float)
for i in firm_ind_u:
	temp1 = np.transpose(np.matrix( lreturns[:,i]))
	[x_cal, y_cal, x_dates] = stock_xy(test_split,fts_space,ws_space, mc_space,news_data,temp1,dates,x_fts, x_ws, x_mc,y[:,i])
	mu_p_ts = np.concatenate([mu_p_ts,mu_news_estimate(x_cal, y_cal, test_split, temp1, dates, n_past,i)],axis=1)
	print(str(datetime.datetime.now())+': Successfully produced mu_p_ts for '+names[i])

#del x_cal, y_cal,news_data, x_fts, x_ws, x_mc, y
gc.collect()


# 7. single stock parameter calibration & get improved cov estimates




# 6. standard past observation past mu and cov
split_point = int(np.floor(np.shape(x_dates)[0]*(1-test_split)))
pmu_p_ts = mu_gen_past(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], n_past)
pcov_p_ts = cov_gen_past(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], n_past)




# 7. build portfolios based on both
[_,first_line] = evaluate_portfolio(names[firm_ind_u],x_dates[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates)
[_,second_line] = evaluate_portfolio(names[firm_ind_u],x_dates[(split_point+1):],lreturns,mu_p_ts,pcov_p_ts,firm_ind_u,dates)
#del dates, names, lreturns, firm_ind_u, x_dates, mu_p_ts, pmu_p_ts, pcov_p_ts, split_point
gc.collect

# 8. plotting the final results
final_plots([first_line,second_line],['standard', 'improved'])


