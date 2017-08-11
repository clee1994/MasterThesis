
import evaluation
import learning

import datetime, pickle, gc, random
import numpy as np


# 0. modifiable variables
path_data = 'Data/'

path_output = 'Output/'
learning.path_output = path_output
evaluation.path_output = path_output

number_jobs = 1
learning.number_jobs = number_jobs
evaluation.number_jobs = number_jobs

past_obs_int = True
learning.past_obs_int = past_obs_int

firms_used = 2
n_past = 80
test_split = 0.35

complet = []

#main function 
def main_x_reg(x_method):
	print(str(datetime.datetime.now())+': Start reading in news:', flush=True)
	news_data = pickle.load(open(path_data + "Reuters.p", "rb" ) )
	print(str(datetime.datetime.now())+': Successfully read all news', flush=True)
	gc.collect()

	if x_method == 4:
		#doc2vec
		[x_gram, dates_news] = learning.calibrate_doc2vec(np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)),dates_prices,test_split,news_data)
		x_tfidf = []
		del news_data
		print(str(datetime.datetime.now())+': Successfully doc2vec', flush=True)
		gc.collect()
	else:
		#uni/bi/tri-grams count/tfidf
		[x_gram, x_tfidf, dates_news] = learning.tfidf_vector(x_method, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split)
		del news_data
		print(str(datetime.datetime.now())+': Successfully '+str(x_method)+'-gram', flush=True)
		gc.collect()
		

	pickle.dump((x_gram, x_tfidf, dates_news), open( path_output+"x_models"+str(x_method)+".p", "wb" ) )

	#[x_gram, x_tfidf, dates_news] = pickle.load(open(path_output +"server_x_models"+str(x_method)+".p", "rb" ) )

	split_point = int(np.floor(np.shape(x_gram[0])[0]*(1-test_split)))
	# gc.collect()

	# #learning.estimate_xgboost, learning.estimate_keras, learning.estimate_linear, learning.estimate_SVR
	# for j in [learning.estimate_ridge,learning.estimate_linear, learning.estimate_SVR]:
	# 	if x_method != 4:
	# 		iter_list = [x_gram,x_tfidf]
	# 	else:
	# 		iter_list = [x_gram]
	# 	for i in iter_list:
	# 		[mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg] = learning.produce_mu_cov(i[0] ,test_split,lreturns, dates_prices, dates_news, n_past, names, firm_ind_u, j)
	# 		[r2,second_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, None, -1)
	# 		[r3,third_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, 0.5, -1)
	# 		complet.append([r2,second_line,r3, third_line, losses, r2m, i[1], parmeters_reg] )
	# 		del mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg, r2,second_line, r3,third_line
	# 		print(str(datetime.datetime.now())+': Successfully learned a vec-reg combination', flush=True)
	# 		gc.collect()
	del x_gram, x_tfidf
	gc.collect()
	return dates_news, split_point


#import multiprocessing
#if __name__ == '__main__':
#	multiprocessing.set_start_method('forkserver')


print(str(datetime.datetime.now())+': Start reading in news:', flush=True)
news_data = pickle.load(open(path_data + "Reuters.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all news', flush=True)
gc.collect()
print(str(datetime.datetime.now())+': Start reading in SP500 data:', flush=True)
[_, dates_prices, names, lreturns] = pickle.load(open(path_data + "SP500.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all data', flush=True)
gc.collect()


#cherry picking -> repair 
#firm_ind_u = learning.sort_predictability(news_data,lreturns,dates_prices,test_split,names)[0:firms_used]
#print(str(datetime.datetime.now())+': Successfully sorted')
#pickle.dump((firm_ind_u), open( path_output + "order.p", "wb" ) )
del news_data
firm_ind_u = pickle.load(open(path_output + "order.p", "rb" ) )
#gc.collect()

#random -> validation, maybe multiple times?
#firm_ind_u = random.sample(range(len(names)-1), firms_used)

#for i in range(4):
#	[dates_news,split_point] = main_x_reg(i+1)
#	gc.collect()
[dates_news, split_point] = main_x_reg(4)

# #benchmark, past obs.
# pmu_p_ts = evaluation.mu_gen_past1(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
# pcov_p_ts = evaluation.cov_gen_past(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
# [r1,first_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, None, -1)
# #[r2,second_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, 0.5, -1)
# #complet.append([r1,first_line,r2, second_line, np.nan, np.nan, ' ','Past observations'] )

# [r4,sp500] = evaluation.pure_SP(dates_news[(split_point+1):],path_data)
# gc.collect()
# evaluation.final_table(complet,np.array(r4),r1,sp500)

# gc.collect()
# pickle.dump((complet,r4,r1,sp500), open( path_output + "final.p", "wb" ) )



