

#plotting
def plot_pred_true_b(y,yhat,benchm,v_m,t_text):
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	import datetime
	import numpy as np


	benchm = np.reshape(benchm,[len(benchm)])
	y = np.reshape(y,[len(y)])
	yhat = np.reshape(yhat,[len(yhat)])

	plt.figure()
	plt.clf()
	#f, axarr = plt.subplots(2, sharex=True)
	plt.plot(y,label= "$y$",linewidth=0.8)
	plt.plot(yhat,label="$\hat{y}_{doc2vec}$",linewidth=0.8)
	plt.plot(benchm,label="$\hat{y}_{past\;obs.}$",linewidth=0.8)
	ts_temp1 = np.abs(np.subtract(yhat,y))
	ts_temp2 = np.abs(np.subtract(benchm,y))
	#axarr[0].text(0.5, 0.5, str(np.sum(np.abs(np.subtract(yhat,y)))), fontdict=font)
	plt.title(t_text)
	#plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3,
	#			ncol=2, mode="expand", borderaxespad=0.,shadow=None,framealpha=1)
	test34 = plt.legend(loc=0,shadow=None,framealpha=1,fancybox=False)
	test34.get_frame().set_edgecolor('black')
	plt.xlabel('Time/Observations')
	plt.ylabel(v_m)
	
	#axarr[1].set_ylabel('Difference to $y$')
	plt.savefig(path_output+'pics/'+str(datetime.datetime.now())+'pred_true.png',bbox_inches='tight',dpi=310)
	plt.close()
	plt.close("all")

	plt.figure()
	plt.clf()
	plt.title(t_text)
	plt.plot(ts_temp1,label="$y - \hat{y}_{doc2vec}$ ($"+str(np.round(np.sum(ts_temp1),4))+"$)",linewidth=0.8)
	plt.plot(ts_temp2,label="$y - \hat{y}_{past\;obs.}$ ($"+str(np.round(np.sum(ts_temp2),4))+"$)",linewidth=0.8)
	#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	#			ncol=2, mode="expand", borderaxespad=0.,shadow=None,framealpha=1)
	test34 = plt.legend(loc=0,shadow=None,framealpha=1,fancybox=False)
	test34.get_frame().set_edgecolor('black')
	plt.xlabel('Time/Observations')
	plt.ylabel('Difference')
	plt.savefig(path_output+'pics/'+str(datetime.datetime.now())+'pred_true_d.png',bbox_inches='tight',dpi=310)
	plt.close()
	plt.close("all")
#evalute portfolio construction

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
	#plt.ylabel('Return')
	plt.savefig(path_output+'pics/'+str(datetime.datetime.now())+'port_performance_ret.png',bbox_inches='tight',dpi=310)
	plt.close()




def learning_curve_plots(grid_results,clf, x_cal, y_cal,n_cpu, alpha_range,gamma_range,show_p):
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)
	import matplotlib as mpl
	import datetime
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	from sklearn.model_selection import learning_curve
	import numpy as np



	train_sizes, train_scores, test_scores = learning_curve(clf, x_cal, y_cal, cv=None, train_sizes=np.linspace(3, len(x_cal)*0.6, 100,dtype=int),scoring='neg_mean_squared_error',n_jobs=-1)
	train_scores = np.mean(train_scores,axis=1)
	test_scores = np.mean(test_scores,axis=1)

	
	plt.figure() 
	plt.clf()

	plt.plot(train_sizes,train_scores*-1, label="Train MSE",linewidth=0.8)
	plt.plot(train_sizes,test_scores*-1, label="Test MSE",linewidth=0.8)
	#plt.xticks(range(len(train_scores)),train_sizes)
	#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,shadow=None,framealpha=1)
	
	test34 = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	           ncol=2, mode="expand", borderaxespad=0.,shadow=None,framealpha=1,fancybox=False)
	test34.get_frame().set_edgecolor('black')
	plt.xlabel('Size training set')
	plt.ylabel('MSE')
	plt.savefig(path_output+'pics/'+str(datetime.datetime.now())+'learning_curve.png',bbox_inches='tight',dpi=310)
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


def final_table(complet, r4,r1):
	#build final table
	f = open(path_output+'tables/final_table.tex', 'w')
	f.write('\\begin{tabular}{ r r r r r r r r r r r r}\n')
	f.write('Vectorization & Regression Model & $\sum$ MSE & $R^2$ & Portfolio & Mean & Variance & Beta & Alpha & Sharpe Ratio & Treynor Ratio & V@R 95 \%  \\\\ \n ')
	f.write('\hline \n')

	[mu, sigma, beta, alpha, sharpe, treynor, var95] = port_measures(r4, r1)
	f.write('\\textit{past obs.} & ' + ' & ' +' & ' + ' & ' + '\\textit{min. var.}' + ' & '+"{:.4f}".format(mu)+' & '+"{:.4f}".format(sigma)+' & '+"{:.4f}".format(beta)+' & '+ "{:.4f}".format(alpha) +' & '+"{:.4f}".format(sharpe)+' & '+"{:.4f}".format(treynor)+' & '+"{:.4f}".format(var95)+'  \\\\ \n ')
			
	for i in complet:
		[mu, sigma, beta, alpha, sharpe, treynor, var95] = port_measures(r4, i[0])
		f.write(i[6] +' & '+ i[7] + ' & '+ "{:.4f}".format(i[4]) +' & '+ "{:.4f}".format(i[5]) + ' & ' + 'min. var.' + ' & '+"{:.4f}".format(mu)+' & '+"{:.4f}".format(sigma)+' & '+"{:.4f}".format(beta)+' & '+ "{:.4f}".format(alpha) +' & '+"{:.4f}".format(sharpe)+' & '+"{:.4f}".format(treynor)+' & '+"{:.4f}".format(var95)+'  \\\\ \n ')
		[mu, sigma, beta, alpha, sharpe, treynor, var95] = port_measures(r4, i[2])
		f.write(' & ' + ' & ' +' & ' + ' & ' + 'min. var. l1' + ' & '+"{:.4f}".format(mu)+' & '+"{:.4f}".format(sigma)+' & '+"{:.4f}".format(beta)+' & '+ "{:.4f}".format(alpha) +' & '+"{:.4f}".format(sharpe)+' & '+"{:.4f}".format(treynor)+' & '+"{:.4f}".format(var95)+'  \\\\ \n ')


	f.write('\\end{tabular}')
	f.close() 


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


