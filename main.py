
import evaluation
import learning

import datetime, pickle, gc, random
import numpy as np


# 0. modifiable variables '/home/ucabjss/Data/'
path_data = 'Data/'

path_output = 'Output/'
learning.path_output = path_output
evaluation.path_output = path_output

number_jobs = 1
learning.number_jobs = number_jobs
evaluation.number_jobs = number_jobs

past_obs_int = True
learning.past_obs_int = past_obs_int

firms_used = 10
n_past = 120
n_past_add = 20
learning.n_past = n_past
learning.n_past_add = n_past_add
n_cov = 5
learning.n_cov = n_cov
test_split = 0.35

complet = []

#main function 
def create_x(x_method,tfidf,lreturns):
	print(str(datetime.datetime.now())+': Start reading in news:', flush=True)
	news_data = pickle.load(open(path_data + "Reuters.p", "rb" ) )
	print(str(datetime.datetime.now())+': Successfully read all news', flush=True)
	gc.collect()

	if x_method == 4:
		#doc2vec
		[documents, y,dates_ret] = learning.create_documents(news_data,np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,learning.data_label_method_val)
		del news_data
		gc.collect()
		print(str(datetime.datetime.now())+': Preprocessed news', flush=True)
		[x_gram, dates_news] = learning.calibrate_doc2vec(documents, y,dates_ret,test_split)
		print(str(datetime.datetime.now())+': Successfully doc2vec', flush=True)
		gc.collect()
	else:
		#uni/bi/tri-grams count/tfidf
		[corpus,x_dates] = learning.create_corpus(x_method, news_data, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split,tfidf)
		del news_data
		print(str(datetime.datetime.now())+': Preprocessed news', flush=True)
		[x_gram, dates_news] = learning.tfidf_vector(x_method, corpus, np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)), dates_prices,test_split,tfidf,x_dates)
		print(str(datetime.datetime.now())+': Successfully '+str(x_method)+'-gram', flush=True)
		gc.collect()
		

	pickle.dump((x_gram, dates_news), open( path_output+ ('bow' if tfidf else 'tfidf') +"x_models"+str(x_method)+".p", "wb" ) )
	split_point = int(np.floor(np.shape(x_gram[0])[0]*(1-test_split)))
	return dates_news,split_point

def reg_x(x_gram,l1,l2,split_point,r1,r4):
	print(str(datetime.datetime.now())+': Start another Regression run on X', flush=True)
	#learning.estimate_xgboost, learning.estimate_keras, learning.estimate_linear, learning.estimate_SVR
	for j in [learning.estimate_ridge,learning.estimate_linear, learning.estimate_SVR]:
		[mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg] = learning.produce_mu_cov(x_gram[0] ,test_split,lreturns, dates_prices, dates_news, n_past, names, firm_ind_u, j)
		[r2,second_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, None, -1)
		[r3,third_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts, cov_p_ts, firm_ind_u,dates_prices,None, 0.5, -1)
		complet.append([r2,second_line,r3, third_line, losses, r2m, x_gram[1], parmeters_reg] )
		evaluation.final_plots([l2,second_line,third_line,l1],[r'past obs.',r'doc2vec',r'doc2vec, l1',r'SP500'])
		pickle.dump((mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg, r2,second_line, r3,third_line), open( path_output +str(datetime.datetime.now())+ "intermediate_save.p", "wb" ) )
		del mu_p_ts, cov_p_ts, losses, r2m, parmeters_reg, r2,second_line, r3,third_line
		print(str(datetime.datetime.now())+': Successfully learned a vec-reg combination', flush=True)
		gc.collect()
		pickle.dump((complet,r4,r1,l1), open( path_output +str(datetime.datetime.now())+ "i_final.p", "wb" ) )



# print(str(datetime.datetime.now())+': Start reading in news:', flush=True)
# news_data = pickle.load(open(path_data + "Reuters.p", "rb" ) )
# print(str(datetime.datetime.now())+': Successfully read all news', flush=True)
# gc.collect()
print(str(datetime.datetime.now())+': Start reading in SP500 data:', flush=True)
[_, dates_prices, names, lreturns] = pickle.load(open(path_data + "SP500.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all data', flush=True)
gc.collect()


#cherry picking -> repair 
# firm_ind_u = learning.sort_predictability(news_data,lreturns,dates_prices,test_split,names)#[0:firms_used]
# print(str(datetime.datetime.now())+': Successfully sorted')
# pickle.dump((firm_ind_u), open( path_output + "order.p", "wb" ) )
# del news_data
# gc.collect()
firm_ind_u = pickle.load(open(path_output + "order.p", "rb" ) )
firm_ind_u = firm_ind_u[0:firms_used]

#random -> validation, maybe multiple times?
#firm_ind_u = random.sample(range(len(names)-1), firms_used)



#create x
# [dates_news, split_point] = create_x(4, True, lreturns)
# gc.collect()
# [dates_news, split_point] = create_x(3, False, lreturns)
# gc.collect()
# [dates_news, split_point] = create_x(3, True, lreturns)
# gc.collect()
# [dates_news, split_point] = create_x(2, False, lreturns)
# gc.collect()
# [dates_news, split_point] = create_x(2, True, lreturns)
# gc.collect()
#[dates_news, split_point] = create_x(1, False, lreturns)
#gc.collect()
#[dates_news, split_point] = create_x(1, True, lreturns)
#gc.collect()

#chanage name to pv dm
[x_gram, dates_news] = pickle.load(open(path_output +"bowx_models4.p", "rb" ) )
split_point = int(np.floor(np.shape(x_gram[0])[0]*(1-test_split)))



#benchmark, past obs.
pmu_p_ts = evaluation.mu_gen_past1(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
pcov_p_ts = evaluation.cov_gen_past(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
[r1,first_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, None, -1)
[r2,second_line] = evaluation.evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, 0.5, -1)
complet.append([r1,first_line,r2, second_line, np.nan, np.nan, ' ','Past observations'] )

[r4,sp500] = evaluation.pure_SP(dates_news[(split_point):],path_data)
gc.collect()


#regression
#reg_x(x_gram,sp500,first_line, split_point,r1,r4)

#change name to pv bdow
[x_gram, dates_news] = pickle.load(open(path_output +"dbow_doc2vec.p", "rb" ) )
#split_point = int(np.floor(np.shape(x_gram[0])[0]*(1-test_split)))
reg_x(x_gram,sp500,first_line, split_point,r1,r4)

[x_gram, dates_news] = pickle.load(open(path_output +"dm_doc2vec.p", "rb" ) )
#split_point = int(np.floor(np.shape(x_gram[0])[0]*(1-test_split)))
reg_x(x_gram,sp500,first_line, split_point,r1,r4)

#[x_gram, dates_news] = pickle.load(open(path_output +"bowx_models1.p", "rb" ) )
#reg_x(x_gram,sp500,first_line, split_point,r1,r4) 

#[x_gram, dates_news] = pickle.load(open(path_output +"bowx_models2.p", "rb" ) )
#reg_x(x_gram,sp500,first_line, split_point,r1,r4)

#[x_gram, dates_news] = pickle.load(open(path_output +"bowx_models3.p", "rb" ) )
#reg_x(x_gram,sp500,first_line, split_point,r1,r4)

#[x_gram, dates_news] = pickle.load(open(path_output +"tfidfx_models1.p", "rb" ) )
#reg_x(x_gram,sp500,first_line, split_point,r1,r4)

#[x_gram, dates_news] = pickle.load(open(path_output +"tfidfx_models2.p", "rb" ) )
#reg_x(x_gram,sp500,first_line, split_point,r1,r4)

#[x_gram, dates_news] = pickle.load(open(path_output +"tfidfx_models3.p", "rb" ) )
#reg_x(x_gram,sp500,first_line, split_point,r1,r4)


# evaluation.final_table(complet,np.array(r4),r1,sp500)

# gc.collect()
pickle.dump((complet,r4,r1,sp500), open( path_output +str(datetime.datetime.now())+  "final.p", "wb" ) )


