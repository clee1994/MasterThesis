
from evaluation import *
from learning import *

from sklearn import svm
from sklearn.linear_model import Ridge

import datetime, pickle, gc, random
import numpy as np


# 0. modifiable variables
firms_used = 5
n_past = 80

path = 'Data/'
#path = '/home/ucabjss/Data/'

#traning splits
test_split = 0.20





# 1. load and preprocess data
print(str(datetime.datetime.now())+': Start reading in news:')
news_data = pickle.load(open(path+ "Reuters.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all news')

print(str(datetime.datetime.now())+': Start reading in SP500 data:')
[_, dates_prices, names, lreturns] = pickle.load(open(path + "SP500.p", "rb" ) )
print(str(datetime.datetime.now())+': Successfully read all data')



# 4. select stocks

#cherry picking
firm_ind_u = sort_predictability(news_data,lreturns,dates_prices,test_split,names,firms_used)[0:firms_used]
print(str(datetime.datetime.now())+': Successfully sorted')

#random -> validation, maybe multiple times?
#firm_ind_u = random.sample(range(len(names)-1), firms_used)



# 3. doc2vec model calibration
[x_doc2vec, dates_news] = calibrate_doc2vec(np.reshape(lreturns[:,firm_ind_u[0]], (np.shape(lreturns)[0],1)),dates_prices,test_split,news_data)
print(str(datetime.datetime.now())+': Successfully calibrated doc2vec model sign')

# 4. tfidf

#create corpus
def tfidf_vector(n_gram, ):
	import numpy as np
	from sklearn.feature_extraction.text import TfidfVectorizer
	import datetime

	corpus = []
	temp_word = ""
	x_dates = []
	prev_d = np.datetime64(datetime.date(1, 1, 1))

	for i in news_data:
		cur_d = np.datetime64(i[0])

		#y
		if cur_d == prev_d:
			prev_d = cur_d
		else:
			if prev_d != np.datetime64(datetime.date(1, 1, 1)):
				#documents.append(TaggedDocument(words,str(doc_c)))
				#doc_c = doc_c + 1
				corpus.append(temp_word)
				temp_word = ""
			x_dates.append(cur_d)
			prev_d = cur_d
		for hj in i[1]:
			temp_word = temp_word + hj + " "
			#words.append(hj)
	corpus.append(temp_word)

	
	vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,n_gram),stop_words='english',min_df=3,max_features=800)
	return vectorizer.fit_transform(corpus).toarray(), x_dates


# unigram

# bigrams

# trigrams


# produce doc2vec ridgeregression -> mu_p_ts
[mu_p_ts, cov_p_ts] = produce_mu_cov(x_calibrated,test_split,lreturns, dates_prices, dates_news, n_past, names, firm_ind_u)




# 6. standard past observation past mu and cov
split_point = int(np.floor(np.shape(x_calibrated)[0]*(1-test_split)))
pmu_p_ts = mu_gen_past1(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
pcov_p_ts = cov_gen_past(lreturns, dates_prices, dates_news[(split_point+1):], firm_ind_u[0:firms_used], n_past)
gc.collect()


# 7. build portfolios based on both
[r1,first_line] = evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,pmu_p_ts,pcov_p_ts,firm_ind_u,dates_prices,None, None, -1)
[r2,second_line] = evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts,cov_p_ts,firm_ind_u,dates_prices,None, None, -1)
[r3,third_line] = evaluate_portfolio(names[firm_ind_u],dates_news[(split_point+1):],lreturns,mu_p_ts,cov_p_ts,firm_ind_u,dates_prices,None, 0.5, -1)
[r4,sp500] = pure_SP(dates_news[(split_point+1):],path)

#del dates, names, lreturns, firm_ind_u, x_dates, mu_p_ts, pmu_p_ts, pcov_p_ts, split_point
gc.collect()


# 8. plotting the final results
final_plots([first_line,second_line,third_line,sp500],[r'min. var. portfolio (past obs.)', r'min. var. portfolio (doc2vec)',r'min. var. (doc2vec, l1)',r'SP500 raw performance'])

#return plots
final_plots_s([r1,r2,r3,r4],[r'past obs.', r'doc2vec',r'doc2vec, l1',r'SP500'])

pickle.dump((first_line,second_line,third_line,sp500,r1,r2,r3,r4), open( "Output/datal.p", "wb" ) )


f = open('Output/tables/'+str(datetime.datetime.now())+'performance.tex', 'w')
f.write('\\begin{tabular}{ l | r r r r r r r}\n')
f.write(' & Mean & Variance & Beta & Alpha & Sharpe Ratio & Treynor Ratio & V@R 95 \%  \\\\ \n ')
f.write('\hline \n')

rets = [r1,r2,r3]
labels = [r'past obs.', r'doc2vec',r'doc2vec, l1']
m_mu = np.mean(r4)
m_sigma = np.var(r4)
r_f = 0.00004
for i in range(3):
	mu = np.mean(rets[i]) 
	sigma = np.var(rets[i])
	beta = np.cov(rets[i],r4)[0,1]/np.var(r4)
	alpha = mu - r_f - beta*(m_mu - r_f)
	sharpe = (mu-r_f)/sigma
	treynor = (mu-r_f)/beta
	var95 = np.percentile(rets[i], 5)
	f.write(labels[i] + ' & '+"{:.4f}".format(mu)+' & '+"{:.4f}".format(sigma)+' & '+"{:.4f}".format(beta)+' & '+ "{:.4f}".format(alpha) +' & '+"{:.4f}".format(sharpe)+' & '+"{:.4f}".format(treynor)+' & '+"{:.4f}".format(var95)+'  \\\\ \n ')

f.write('\\end{tabular}')
f.close() 

#assuming some risk free rate


