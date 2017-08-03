
from evaluation import *
from learning import *

from sklearn import svm
from sklearn.linear_model import Ridge

import datetime, pickle, gc, random
import numpy as np


# 0. modifiable variables
firms_used = 2
n_past = 80

path = 'Data/'
#path = '/home/ucabjss/Data/'

#traning splits
test_split = 0.20


print(str(datetime.datetime.now())+': Start reading in news:')
news_data = pickle.load(open(path+ "Reuters.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all news')

print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[_, dates_prices, names, lreturns] = pickle.load(open(path + "SP500.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all data')



#cherry picking -> repair 
firm_ind_u = sort_predictability(news_data,lreturns,dates_prices,test_split,names)[0:firms_used]
print(str(datetime.datetime.now())+': Successfully sorted')


#random -> validation, maybe multiple times?
#firm_ind_u = random.sample(range(len(names)-1), firms_used)


# unigram count/tfidf
[x_unigram_count, x_unigram_tfidf, dates_news] = tfidf_vector(1, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)

# bigrams count/tfidf
[x_bigram_count, x_bigram_tfidf, dates_news] = tfidf_vector(2, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)

# trigrams count/tfidf
[x_trigram_count, x_trigram_tfidf, dates_news] = tfidf_vector(3, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)

# doc2vec
[x_doc2vec, dates_news] = calibrate_doc2vec(np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)),dates_prices,test_split,news_data)


# 6. standard past observation past mu and cov
split_point = int(np.floor(np.shape(x_doc2vec[0])[0]*(1-test_split)))
pmu_p_ts = mu_gen_past1(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
pcov_p_ts = cov_gen_past(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)


complet = []
for j in [estimate_linear, estimate_ridge, estimate_SVR, estimate_xgboost, estimate_keras]:
	[r1,first_line] = evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, None, -1)
	for i in [x_unigram_count, x_unigram_tfidf, x_bigram_count, x_bigram_tfidf, x_trigram_count, x_trigram_tfidf, x_doc2vec]:
		[mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg] = produce_mu_cov(i[0] ,test_split,lreturns, dates_prices, dates_news, n_past, names, firm_ind_u, j)
		[r2,second_line] = evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, None, -1)
		[r3,third_line] = evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, 0.5, -1)
		complet.append([r2,second_line,r3, third_line, r1, first_line, losses, r2m,i[1], parmeters_reg] )


[r4,sp500] = pure_SP(dates_news[(split_point+1):],path)

final_table(complet,r4)


# final_plots([first_line,second_line,third_line,sp500],[r'min. var. portfolio (past obs.)', r'min. var. portfolio (doc2vec)',r'min. var. (doc2vec, l1)',r'SP500 raw performance'])

# #return plots
# final_plots_s([r1,r2,r3,r4],[r'past obs.', r'doc2vec',r'doc2vec, l1',r'SP500'])

# pickle.dump((first_line,second_line,third_line,sp500,r1,r2,r3,r4), open( "Output/datal.p", "wb" ) )




