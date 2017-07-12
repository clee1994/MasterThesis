#plotting
def plot_pred_true(y,yhat):
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	import datetime
	import numpy as np
	#print(np.shape(benchm))


	plt.figure()
	plt.clf()
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(y,label= "true y")
	axarr[0].plot(yhat,label="predicted y")
	axarr[1].plot(np.abs(np.subtract(yhat,y)),label="difference prediction and true")
	axarr[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	   			ncol=2, mode="expand", borderaxespad=0.)
	axarr[0].xlabel('Time/Observations')
	axarr[0].ylabel('Mean/Volatility')
	axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	   			ncol=2, mode="expand", borderaxespad=0.)
	axarr[1].xlabel('Time/Observations')
	axarr[1].ylabel('Difference of true and estimated mean/volatility')
	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'pred_true.png',bbox_inches='tight')
	plt.close()
	plt.close("all")

#plotting
def plot_pred_true_b(y,yhat,benchm,v_m):
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
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(y,label= "$y$")
	axarr[0].plot(yhat,label="$\hat{y}_{doc2vec}$")
	axarr[0].plot(benchm,label="$\hat{y}_{past\;obs.}$")
	ts_temp1 = np.abs(np.subtract(yhat,y))
	ts_temp2 = np.abs(np.subtract(benchm,y))
	#axarr[0].text(0.5, 0.5, str(np.sum(np.abs(np.subtract(yhat,y)))), fontdict=font)
	axarr[1].plot(ts_temp1,label="$y - \hat{y}_{doc2vec}$ ($"+str(np.round(np.sum(ts_temp1),4))+"$)")
	axarr[1].plot(ts_temp2,label="$y - \hat{y}_{past\;obs.}$ ($"+str(np.round(np.sum(ts_temp2),4))+"$)")
	axarr[0].legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3,
				ncol=2, mode="expand", borderaxespad=0.)
	axarr[0].set_xlabel('Time/Observations')
	axarr[0].set_ylabel(v_m)
	axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
				ncol=2, mode="expand", borderaxespad=0.)
	axarr[1].set_xlabel('Time/Observations')
	#axarr[1].set_ylabel('Difference to $y$')
	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'pred_true.png',bbox_inches='tight')
	plt.close()
	plt.close("all")

#evalute portfolio construction

def mu_gen_past(lreturns, dates_lr, x_dates_test, used_stocks_ind, n_past):
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

		mu.append(np.mean(lreturns[(ind_d-(n_past+1)):(ind_d-1),used_stocks_ind],axis=0))

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


def evaluate_portfolio(used_stocks,x_dates_test,lreturns,mu_ts,cov_ts,firm_ind,dates):
	from port_opt import min_var_mu, min_var, ret2prices
	import datetime
	import numpy as np

	# firm_ind = list()
	# for i in range(firms_used):
	# 	firm_ind.append(list(names).index(used_stocks[i]))

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

		mu = mu_ts[i]#np.nanmean(lreturns[(ind_d-n_past):ind_d, firm_ind],axis=0)

		#depends what shall be tested
		#improved_mu = np.zeros(firms_used)
		#using news to improve
		#for j in range(firms_used):
			#absolutly ridiculus change
			#temp_data_input_predict = np.reshape(x_test[i,:], (1,) + x_test[i,:].shape)
			#mu_change = predict_y[j][i]
			#[j].predict(temp_data_input_predict)
			#improved_mu[j] = mu[j] + mu_change

		gamma = cov_ts[i]#np.cov(lreturns[(ind_d-n_past):ind_d, firm_ind],rowvar=False)

		[w, mu_p, var_p] = min_var(mu, gamma)
		#[i_w, i_mu_p, i_var_p] = min_var(improved_mu, gamma)
		#the plus here, also not really sure
		realized_mu.append(np.dot(w,lreturns[ind_d+1,firm_ind]))
		#i_realized_mu.append(np.dot(i_w,lreturns[ind_d,firm_ind]))


	#visualization of results!
	realized_mu = np.array(realized_mu).flatten()
	value_over_time = ret2prices(realized_mu,100)

	#i_realized_mu = np.array(i_realized_mu).flatten()
	#i_value_over_time = ret2prices(i_realized_mu,100)


	# import matplotlib as mpl
	# mpl.use('Agg')
	# import matplotlib.pyplot as plt
	# plt.figure() 
	# plt.clf()
	# plt.plot(value_over_time , 'r', label='ordinary min var')
	# #plt.plot(i_value_over_time , 'b', label='improved min var portfolio')

	# #plt.plot(np.subtract(value_over_time,i_value_over_time), 'g', label='improved min var portfolio')
	# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	#            ncol=2, mode="expand", borderaxespad=0.)

	# plt.savefig('Output/pics/'+str(datetime.datetime.now())+'port_performance.png')
	# plt.close()

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
		plt.plot(arg_lines[i], label=label_list[i])
	#plt.plot(second_line , 'b', label='improved min var portfolio')

	#plt.plot(np.subtract(value_over_time,i_value_over_time), 'g', label='improved min var portfolio')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	           ncol=2, mode="expand", borderaxespad=0.)
	plt.xlabel('Time/Observations')
	plt.ylabel('Value/USD')

	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'port_performance.png',bbox_inches='tight')
	plt.close()


def learning_plots(grid_results,clf, x_cal, y_cal,n_cpu, alpha_range,gamma_range):
	#plot grid results
	from matplotlib import rc
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=True)
	import matplotlib as mpl
	import datetime
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from sklearn.model_selection import learning_curve
	import numpy as np



	index_linear = np.where(grid_results['param_kernel']=='linear')
	val_train = np.array(grid_results['mean_train_score'])[index_linear]*-1
	val_test = np.array(grid_results['mean_test_score'])[index_linear]*-1
	plt.figure() 
	plt.clf()
	plt.plot(val_train, label="Train MSE")
	plt.plot(val_test, label="Test MSE")
	plt.xticks(range(len(val_train)),np.array(grid_results['param_alpha'])[index_linear])
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
	plt.xlabel('Alpha')
	plt.ylabel('MSE')

	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'alpha_fitting.png',bbox_inches='tight')
	plt.close()



	index_linear = np.where(grid_results['param_kernel']=='rbf')

	val_train = np.reshape(np.array(grid_results['mean_train_score'])[index_linear]*-1,[len(alpha_range),len(gamma_range)])
	val_test = np.reshape(np.array(grid_results['mean_test_score'])[index_linear]*-1,[len(alpha_range),len(gamma_range)])
	x_pval = np.reshape(np.array(grid_results['param_alpha'])[index_linear],[len(alpha_range),len(gamma_range)])
	y_pval = np.reshape(np.array(grid_results['param_gamma'])[index_linear],[len(alpha_range),len(gamma_range)])
	

	
	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(x_pval, y_pval, val_test)


	#plt.plot(val_train, label="Train MSE")
	#plt.plot(val_test, label="Test MSE")
	#plt.xticks(range(len(val_train)),np.array(grid_results['param_alpha'])[index_linear])
	#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
	ax.set_xlabel('Alpha')
	ax.set_ylabel('Gamma')
	ax.set_zlabel('MSE')

	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'rbf_fitting.png',bbox_inches='tight')
	plt.close()


	

	train_sizes, train_scores, test_scores = learning_curve(clf, x_cal, y_cal, cv=None, train_sizes=np.linspace(3, len(x_cal)*0.6, 100,dtype=int),n_jobs=n_cpu,scoring='neg_mean_squared_error')
	train_scores = np.mean(train_scores,axis=1)
	test_scores = np.mean(test_scores,axis=1)

	
	plt.figure() 
	plt.clf()

	plt.plot(train_sizes,train_scores*-1, label="Train MSE")
	plt.plot(train_sizes,test_scores*-1, label="Test MSE")
	#plt.xticks(range(len(train_scores)),train_sizes)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
	plt.xlabel('Size training set')
	plt.ylabel('MSE')

	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'learning_curve.png',bbox_inches='tight')
	plt.close()
