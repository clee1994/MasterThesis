
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

complet = []

#main function 
def main_x_reg(x_method):
	print(str(datetime.datetime.now())+': Start reading in news:')
	news_data = pickle.load(open(path_data + "Reuters.p", "rb" ) )
	print(str(datetime.datetime.now())+': Successfully read all news')
	gc.collect()

	if x_method == 0:
		#unigram count/tfidf
		[x_gram, x_tfidf, dates_news] = learning.tfidf_vector(1, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)
		del news_data
		print(str(datetime.datetime.now())+': Successfully 1-gram')
		gc.collect()

	elif x_method == 1:
		#bigrams count/tfidf
		[x_gram, x_tfidf, dates_news] = learning.tfidf_vector(2, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)
		del news_data
		print(str(datetime.datetime.now())+': Successfully 2-gram')
		gc.collect()
	elif x_method == 2:
		#trigrams count/tfidf
		[x_gram, x_tfidf, dates_news] = learning.tfidf_vector(3, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)
		print(str(datetime.datetime.now())+': Successfully 3-gram')
		gc.collect()
	else:
		# doc2vec
		[x_gram, dates_news] = learning.calibrate_doc2vec(np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)),dates_prices,test_split,news_data)
		del news_data
		print(str(datetime.datetime.now())+': Successfully doc2vec')
		gc.collect()
	#x_unigram_count, x_unigram_tfidf, x_bigram_count, x_bigram_tfidf, x_trigram_count, x_trigram_tfidf, 
	pickle.dump((x_gram, x_tfidf, dates_news), open( path_output+"x_models"+str(x_method)+".p", "wb" ) )

	split_point = int(np.floor(np.shape(x_gram[0])[0]*(1-test_split)))
	pmu_p_ts = evaluation.mu_gen_past1(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
	pcov_p_ts = evaluation.cov_gen_past(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
	gc.collect()

	complet = []
	#learning.estimate_xgboost, learning.estimate_keras
	#x_unigram_count, x_unigram_tfidf, x_bigram_count, x_bigram_tfidf, x_trigram_count, x_trigram_tfidf,
	for j in [learning.estimate_linear, learning.estimate_ridge, learning.estimate_SVR]:
		[r1,first_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, None, -1)
		if x_method != 3:
			for i in [x_gram,x_tfidf]:
				[mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg] = learning.produce_mu_cov(i[0] ,test_split,lreturns, dates_prices, dates_news, n_past, names, firm_ind_u, j)
				[r2,second_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, None, -1)
				[r3,third_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, 0.5, -1)
				complet.append([r2,second_line,r3, third_line, r1, first_line, losses, r2m,i[1], parmeters_reg] )
				del mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg, r2,second_line, r3,third_line
				print(str(datetime.datetime.now())+': Successfully learned a vec-reg combination')
				gc.collect()
		else:
			for i in [x_gram]:
				[mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg] = learning.produce_mu_cov(i[0] ,test_split,lreturns, dates_prices, dates_news, n_past, names, firm_ind_u, j)
				[r2,second_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, None, -1)
				[r3,third_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, 0.5, -1)
				complet.append([r2,second_line,r3, third_line, r1, first_line, losses, r2m,i[1], parmeters_reg] )
				del mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg, r2,second_line, r3,third_line
				print(str(datetime.datetime.now())+': Successfully learned a vec-reg combination')
				gc.collect()


import multiprocessing
if __name__ == '__main__':
	multiprocessing.set_start_method('forkserver')


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
	del news_data
	gc.collect()

	#random -> validation, maybe multiple times?
	#firm_ind_u = random.sample(range(len(names)-1), firms_used)

	for i in range(4):
		main_x_reg(i)

	[r4,sp500] = evaluation.pure_SP(dates_news[(split_point+1):],path_data)
	gc.collect()
	evaluation.final_table(complet,r4)
	gc.collect()
	pickle.dump((complet,r4, sp500), open( path_output + "final.p", "wb" ) )

	#plot the best way to do it -> pred/true, learning curve, final curves

	# final_plots([first_line,second_line,third_line,sp500],[r'min. var. portfolio (past obs.)', r'min. var. portfolio (doc2vec)',r'min. var. (doc2vec, l1)',r'SP500 raw performance'])
	# final_plots_s([r1,r2,r3,r4],[r'past obs.', r'doc2vec',r'doc2vec, l1',r'SP500'])




