
import evaluation
import learning

import datetime, pickle, gc, random
import numpy as np


# 0. modifiable variables
firms_used = 2
n_past = 80

path_data = 'Data/'
#path_data = '/home/ucabjss/Data/'

path_output = 'Output/'
#path_output = '/home/ucabjss/Scratch/Output/'
learning.path_output = path_output
evaluation.path_output = path_output

#traning splits
test_split = 0.55


print(str(datetime.datetime.now())+': Start reading in news:')
news_data = pickle.load(open(path_data + "Reuters.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all news')
gc.collect()
print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[_, dates_prices, names, lreturns] = pickle.load(open(path_data + "SP500.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all data')
gc.collect()


#cherry picking -> repair 
firm_ind_u = learning.sort_predictability(news_data,lreturns,dates_prices,test_split,names)[0:firms_used]
print(str(datetime.datetime.now())+': Successfully sorted')
gc.collect()

#random -> validation, maybe multiple times?
#firm_ind_u = random.sample(range(len(names)-1), firms_used)


# unigram count/tfidf
[x_unigram_count, x_unigram_tfidf, dates_news] = learning.tfidf_vector(1, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)
print(str(datetime.datetime.now())+': Successfully 1-gram')
gc.collect()
# bigrams count/tfidf
[x_bigram_count, x_bigram_tfidf, dates_news] = learning.tfidf_vector(2, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)
print(str(datetime.datetime.now())+': Successfully 2-gram')
gc.collect()
# trigrams count/tfidf
[x_trigram_count, x_trigram_tfidf, dates_news] = learning.tfidf_vector(3, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)
print(str(datetime.datetime.now())+': Successfully 3-gram')
gc.collect()
# doc2vec
[x_doc2vec, dates_news] = learning.calibrate_doc2vec(np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)),dates_prices,test_split,news_data)
del news_data
print(str(datetime.datetime.now())+': Successfully doc2vec')
gc.collect()
pickle.dump((x_unigram_count, x_unigram_tfidf, x_bigram_count, x_bigram_tfidf, x_trigram_count, x_trigram_tfidf, x_doc2vec, dates_news), open( path_output+"x_models.p", "wb" ) )

split_point = int(np.floor(np.shape(x_doc2vec[0])[0]*(1-test_split)))
pmu_p_ts = evaluation.mu_gen_past1(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
pcov_p_ts = evaluation.cov_gen_past(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
gc.collect()

complet = []
#learning.estimate_xgboost, learning.estimate_keras
for j in [learning.estimate_linear, learning.estimate_ridge, learning.estimate_SVR]:
	[r1,first_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, None, -1)
	for i in [x_unigram_count, x_unigram_tfidf, x_bigram_count, x_bigram_tfidf, x_trigram_count, x_trigram_tfidf, x_doc2vec]:
		[mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg] = learning.produce_mu_cov(i[0] ,test_split,lreturns, dates_prices, dates_news, n_past, names, firm_ind_u, j)
		[r2,second_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, None, -1)
		[r3,third_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, 0.5, -1)
		complet.append([r2,second_line,r3, third_line, r1, first_line, losses, r2m,i[1], parmeters_reg] )
		print(str(datetime.datetime.now())+': Successfully learned a vec-reg combination')
		gc.collect()

[r4,sp500] = evaluation.pure_SP(dates_news[(split_point+1):],path_data)
gc.collect()
evaluation.final_table(complet,r4)
gc.collect()
pickle.dump((complet,r4, sp500), open( path_output + "final.p", "wb" ) )

#plot the best way to do it -> pred/true, learning curve, final curves

# final_plots([first_line,second_line,third_line,sp500],[r'min. var. portfolio (past obs.)', r'min. var. portfolio (doc2vec)',r'min. var. (doc2vec, l1)',r'SP500 raw performance'])
# final_plots_s([r1,r2,r3,r4],[r'past obs.', r'doc2vec',r'doc2vec, l1',r'SP500'])



