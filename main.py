from data_loading import load_news_data, load_SP_data 
from learning import * #gen_xy_daily, train_test_split
from evaluation import *
from stocks.stocks_big import stocks_used
import numpy as np
import datetime 
import pickle


#modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
firms_used = 25

#traning splits
test_split = 0.15

#doc2vec spaces
#fts_space = np.linspace(180,440,8,dtype=int)
#ws_space = np.linspace(4,18,8,dtype=int)
#mc_space = np.linspace(0,35,8,dtype=int)

fts_space = np.linspace(150,550,12,dtype=int)
ws_space = np.linspace(2,22,12,dtype=int)
mc_space = np.linspace(0,50,12,dtype=int)


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
[x_cal, y_cal, x_dates] = stock_xy(firms_used,test_split, firm_ind_u,fts_space,ws_space, mc_space,news_data,lreturns,dates)

#get improved mu estimates
mu_p_ts = mu_news_estimate(x_cal, y_cal, test_split, lreturns, firms_used, firm_ind_u,dates)


#gen mu ts
split_point = int(np.floor(np.shape(x_dates)[0]*(1-test_split)))

pmu_p_ts = mu_gen_past(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], 50)


#gen cov ts
pcov_p_ts = cov_gen_past(lreturns, dates, x_dates[(split_point+1):], firm_ind_u[0:firms_used], 50)


#plug it into portfolio


[_,first_line] = evaluate_portfolio(np.array(names)[firm_ind_u[0:firms_used]],x_dates[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u[0:firms_used],dates)
[_,second_line] = evaluate_portfolio(np.array(names)[firm_ind_u[0:firms_used]],x_dates[(split_point+1):],lreturns,mu_p_ts,pcov_p_ts,firm_ind_u[0:firms_used],dates)



#plotting the final results
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.figure() 
plt.clf()
plt.plot(first_line , 'r', label='ordinary min var')
plt.plot(second_line , 'b', label='improved min var portfolio')

#plt.plot(np.subtract(value_over_time,i_value_over_time), 'g', label='improved min var portfolio')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.savefig('Output/pics/'+str(datetime.datetime.now())+'port_performance.png')
plt.close()