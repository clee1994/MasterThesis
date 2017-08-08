
#plotting
def plot_pred_true_b(grid_results,clf, x_cal, y_cal,n_cpu, alpha_range,gamma_range, y,yhat,benchm,v_m,t_text):
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	import datetime
	import numpy as np
	from sklearn.model_selection import learning_curve


	benchm = np.reshape(benchm,[len(benchm)])
	y = np.reshape(y,[len(y)])
	yhat = np.reshape(yhat,[len(yhat)])

	train_sizes, train_scores, test_scores = learning_curve(clf, x_cal, y_cal, cv=None, train_sizes=np.linspace(3, len(x_cal)*0.6, 100,dtype=int),scoring='neg_mean_squared_error',n_jobs=number_jobs)
	train_scores = np.mean(train_scores,axis=1)
	test_scores = np.mean(test_scores,axis=1)

	ts_temp1 = np.abs(np.subtract(yhat,y))
	ts_temp2 = np.abs(np.subtract(benchm,y))


	fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(35,10))


	ax[0].plot(y,label= "$y$",linewidth=0.8)
	ax[0].plot(yhat,label="$\hat{y}_{doc2vec}$",linewidth=1)
	ax[0].plot(benchm,label="$\hat{y}_{past\;obs.}$",linewidth=1)
	test34 = ax[0].legend(loc=1,shadow=None,framealpha=1,fancybox=False)
	test34.get_frame().set_edgecolor('black')
	ax[0].set_xlabel('Time/Observations')
	ax[0].set_ylabel(v_m)



	ax[1].plot(ts_temp1,label="$y - \hat{y}_{doc2vec}$ ($"+str(np.round(np.sum(ts_temp1),4))+"$)",linewidth=1)
	ax[1].plot(ts_temp2,label="$y - \hat{y}_{past\;obs.}$ ($"+str(np.round(np.sum(ts_temp2),4))+"$)",linewidth=1)
	test34 = ax[1].legend(loc=1,shadow=None,framealpha=1,fancybox=False)
	test34.get_frame().set_edgecolor('black')
	ax[1].set_xlabel('Time/Observations')
	ax[1].set_ylabel('Difference')


	ax[2].plot(train_sizes,train_scores*-1, label="Train MSE",linewidth=1)
	ax[2].plot(train_sizes,test_scores*-1, label="Test MSE",linewidth=1)
	test34 = ax[2].legend(loc=1, shadow=None,framealpha=1,fancybox=False)
	test34.get_frame().set_edgecolor('black')
	ax[2].set_xlabel('Size training set')
	ax[2].set_ylabel('MSE')


	plt.savefig(path_output+'pics/'+str(datetime.datetime.now())+'learning_curve.png',bbox_inches='tight',dpi=310)
	plt.close()

def mu_gen_past1(lreturns, dates_lr, x_dates_test, used_stocks_ind, n_past):
	import numpy as np
	import datetime

	mu = []
	for i in range(len(x_dates_test)):
		temp = x_dates_test[i].tolist()
		cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
		try:
			ind_d = list(dates_lr).index(cur_d)
		except:
			temp1 = min(dates_lr, key=lambda x: abs(x - cur_d))
			ind_d = list(dates_lr).index(temp1)
		mu.append(np.nanmean(lreturns[(ind_d-(n_past+1)):(ind_d-1),used_stocks_ind],axis=0))

	mu = np.array(mu)
	return mu

def cov_gen_past(lreturns, dates_lr, x_dates_test, used_stocks_ind, n_past):
	import numpy as np
	import datetime

	mu = []
	for i in range(len(x_dates_test)):
		temp = x_dates_test[i].tolist()
		cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
		try:
			ind_d = list(dates_lr).index(cur_d)
		except:
			temp1 = min(dates_lr, key=lambda x: abs(x - cur_d))
			ind_d = list(dates_lr).index(temp1)

		mu.append(np.cov(lreturns[(ind_d-(n_past+1)):(ind_d-1),used_stocks_ind],rowvar=False))

	mu = np.array(mu)
	return mu


def evaluate_portfolio(used_stocks,x_dates_test,lreturns,mu_ts,cov_ts,firm_ind,dates, e_mu, glambda, h):
	from port_opt import cv_opt, ret2prices
	import datetime
	import numpy as np

	realized_mu = list()
	i_realized_mu = list()

	for i in range(len(x_dates_test)):
		temp = x_dates_test[i].tolist()
		cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
		try:
			ind_d = list(dates).index(cur_d)
		except:
			temp1 = min(dates, key=lambda x: abs(x - cur_d))
			ind_d = list(dates).index(temp1)

		mu = mu_ts[i]

		gamma = cov_ts[i]

		[w, mu_p, var_p] = cv_opt(mu, gamma, e_mu, glambda, h)

		realized_mu.append(np.dot(np.transpose(w),lreturns[ind_d+1,firm_ind]))


	#visualization of results!
	realized_mu = np.array(realized_mu).flatten()
	value_over_time = ret2prices(realized_mu,100)

	return realized_mu, value_over_time


def final_plots(arg_lines,label_list):
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	import datetime
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt

	plt.figure() 
	plt.clf()
	for i in range(len(arg_lines)):
		plt.plot(arg_lines[i], label=label_list[i],linewidth=0.8)

	test34 = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	           ncol=2, mode="expand", borderaxespad=0.,shadow=None,framealpha=1,fancybox=False)
	test34.get_frame().set_edgecolor('black')
	plt.xlabel('Time/Observations')
	plt.ylabel('Value/USD')
	plt.savefig(path_output+'pics/'+str(datetime.datetime.now())+'port_performance.png',bbox_inches='tight',dpi=310)
	plt.close()



def final_plots_s(arg_lines,label_list):
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	import datetime
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt

	plt.figure() 
	plt.clf()
	f, axarr = plt.subplots(len(arg_lines), sharex=True)
	for i in range(len(arg_lines)):
		axarr[i].plot(arg_lines[i], label=label_list[i],linewidth=0.8)
		axarr[i].set_ylabel(label_list[i])
	plt.xlabel('Time/Observations')
	plt.savefig(path_output+'pics/'+str(datetime.datetime.now())+'port_performance_ret.png',bbox_inches='tight',dpi=310)
	plt.close()







def pure_SP(x_dates, path):
	import pandas as pd
	import numpy as np
	import datetime
	from port_opt import ret2prices

	raw_data = pd.read_csv(path + 'pureSP500.csv', sep=',',header=None,low_memory=False)

	prices = raw_data.values[1:,5].astype(float)
	dates = np.array(raw_data.values[1:,0],dtype='datetime64')
	lreturns = np.diff(np.log(prices),n=1, axis=0)

	ret = []
	for i in range(len(x_dates)):
		temp = x_dates[i].tolist()
		cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
		try:
			ind_d = list(dates).index(cur_d)
		except:
			temp1 = min(dates, key=lambda x: abs(x - cur_d))
			ind_d = list(dates).index(temp1)
		ret.append(lreturns[ind_d])

	return ret,np.array(ret2prices(ret,100))


def final_table(complet, r4,r1,sp500):
	from port_opt import ret2prices
	import numpy as np

	#build final table
	f = open(path_output+'tables/final_table.tex', 'w')
	f.write('\\begin{tabular}{ r r r r r r r r r r r r}\n')
	f.write('Vectorization & Regression Model & $\sum$ MSE & $R^2$ & Portfolio & Mean & Variance & Beta & Alpha & Sharpe Ratio & Treynor Ratio & V@R 95 \%  \\\\ \n ')
	f.write('\hline \n')


	order = np.argsort(np.array(complet)[:,4])
	[mu, sigma, beta, alpha, sharpe, treynor, var95] = port_measures(r4, r1)
	f.write('\\textit{past obs.} & ' + ' & ' +' & ' + ' & ' + '\\textit{min. var.}' + ' & '+"{:.4f}".format(mu)+' & '+"{:.4f}".format(sigma)+' & '+"{:.4f}".format(beta)+' & '+ "{:.4f}".format(alpha) +' & '+"{:.4f}".format(sharpe)+' & '+"{:.4f}".format(treynor)+' & '+"{:.4f}".format(var95)+'  \\\\ \n ')
			
	for i in order:
		[mu, sigma, beta, alpha, sharpe, treynor, var95] = port_measures(r4, complet[i][0])
		f.write(complet[i][6] +' & '+ complet[i][7] + ' & '+ "{:.4f}".format(complet[i][4]) +' & '+ "{:.4f}".format(complet[i][5]) + ' & ' + 'min. var.' + ' & '+"{:.4f}".format(mu)+' & '+"{:.4f}".format(sigma)+' & '+"{:.4f}".format(beta)+' & '+ "{:.4f}".format(alpha) +' & '+"{:.4f}".format(sharpe)+' & '+"{:.4f}".format(treynor)+' & '+"{:.4f}".format(var95)+'  \\\\ \n ')
		[mu, sigma, beta, alpha, sharpe, treynor, var95] = port_measures(r4, complet[i][2])
		f.write(' & ' + ' & ' +' & ' + ' & ' + 'min. var. l1' + ' & '+"{:.4f}".format(mu)+' & '+"{:.4f}".format(sigma)+' & '+"{:.4f}".format(beta)+' & '+ "{:.4f}".format(alpha) +' & '+"{:.4f}".format(sharpe)+' & '+"{:.4f}".format(treynor)+' & '+"{:.4f}".format(var95)+'  \\\\ \n ')


	f.write('\\end{tabular}')
	f.close() 

	final_plots_s([r1,complet[order[0]][0],complet[order[0]][2],r4],[r'past obs.',r'doc2vec',r'doc2vec, l1',r'SP500'])
	final_plots([ret2prices(r1,100),complet[order[0]][1],complet[order[0]][3],sp500],[r'past obs.',r'doc2vec',r'doc2vec, l1',r'SP500'])


def port_measures(rbase, ret):
	import numpy as np

	m_mu = np.mean(rbase)
	m_sigma = np.var(rbase)
	r_f = 0.00004 #assuming some risk free rate
	mu = np.mean(ret) 
	sigma = np.var(ret)
	beta = np.cov(ret,rbase)[0,1]/np.var(rbase)
	alpha = mu - r_f - beta*(m_mu - r_f)
	sharpe = (mu-r_f)/sigma
	treynor = (mu-r_f)/beta
	var95 = np.percentile(ret, 5)
	return mu, sigma, beta, alpha, sharpe, treynor, var95

def doc2vec_tables(model,documents):
	import datetime
	import numpy as np

	doc_id = np.random.randint(model.docvecs.count)
	sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
	target = ' '.join(documents[doc_id].words)
	closest = ' '.join(documents[int(sims[0][0])].words)
	least = ' '.join(documents[int(sims[len(sims) - 1][0])].words)

	chars_pl = 65



	f = open(path_output+'tables/'+str(datetime.datetime.now())+'target.tex', 'w')
	f.write('"'+ target[0:(chars_pl-1)] + '\n')

	for i in range(9):
		f.write(target[(i+1)*(chars_pl-1):(i+2)*(chars_pl-1)] + '\n')
	f.write('... \n')

	for i in np.arange(9,0,-1):
		f.write(target[-(i+2)*chars_pl:-(i+1)*chars_pl] + '\n')
	f.write(target[-chars_pl:-1]+'"\n')
	f.write('Date: '+ str(x_dates[doc_id].astype('M8[D]')) + '\n')
	f.write('Number of characters: ' + str(len(target)) + '\n')
	f.close() 


	f = open(path_output+'tables/'+str(datetime.datetime.now())+'closest.tex', 'w')
	f.write('"'+ closest[0:(chars_pl-1)]+ '\n')

	for i in range(9):
		f.write(closest[(i+1)*(chars_pl-1):(i+2)*(chars_pl-1)] + '\n')
	f.write('... \n')

	for i in np.arange(9,0,-1):
		f.write(closest[-(i+2)*chars_pl:-(i+1)*chars_pl] + '\n')
	f.write(closest[-chars_pl:-1]+'"\n')
	f.write('Date: '+ str(x_dates[int(sims[0][0])].astype('M8[D]')) + '\n')
	f.write('Number of characters: ' + str(len(closest)) + '\n')
	f.close() 

	f = open(path_output+'tables/'+str(datetime.datetime.now())+'least.tex', 'w')
	f.write('"'+ least[0:(chars_pl-1)]+ '\n')

	for i in range(9):
		f.write(least[(i+1)*(chars_pl-1):(i+2)*(chars_pl-1)] + '\n')
	f.write('... \n')

	for i in np.arange(9,0,-1):
		f.write(least[-(i+2)*chars_pl:-(i+1)*chars_pl] + '\n')
	f.write(least[-chars_pl:-1]+'"\n')
	f.write('Date: '+ str(x_dates[int(sims[len(sims) - 1][0])].astype('M8[D]')) + '\n')
	f.write('Number of characters: ' + str(len(least)) + '\n')
	f.close()

	#words -> actually not of relevance but cool to see
	import random
	exword = random.choice(model.wv.index2word)
	similars_words = str(model.most_similar(exword, topn=20)).replace('), ',')\n')

	f = open(path_output+'tables/'+str(datetime.datetime.now())+'wordss.tex', 'w')
	f.write('"'+exword + '"\n')
	f.write(similars_words)
	f.close() 

def make_pred_sort_table(firm_ind_u, loss, names):
	import numpy as np

	f = open(path_output+'tables/pred_sort.tex', 'w')
	f.write('\\begin{tabular}{ r | l }\n')
	f.write('Stock Ticker & MSE \\\\ \n ')
	f.write('\hline \n')
	for i in range(10):
		f.write(names[firm_ind_u[i]]+' & '+ "{:.4f}".format((loss[i]))+' \\\\ \n ')
	f.write('\hline \n')
	f.write('\hline \n')
	f.write('Mean & '+ "{:.4f}".format(np.nanmean(loss[:]))+' \\\\ \n ')
	f.write('\\end{tabular}')
	f.close() 
