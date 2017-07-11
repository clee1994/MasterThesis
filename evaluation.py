#plotting
def plot_pred_true(y,yhat):
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
	axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	   			ncol=2, mode="expand", borderaxespad=0.)
	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'pred_true.png',bbox_inches='tight')
	plt.close()
	plt.close("all")

#plotting
def plot_pred_true_b(y,yhat,benchm):
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
	axarr[0].plot(y,label= "true y")
	axarr[0].plot(yhat,label="predicted y")
	axarr[0].plot(benchm,label="benchmark")
	ts_temp1 = np.abs(np.subtract(yhat,y))
	ts_temp2 = np.abs(np.subtract(benchm,y))
	#axarr[0].text(0.5, 0.5, str(np.sum(np.abs(np.subtract(yhat,y)))), fontdict=font)
	axarr[1].plot(ts_temp1,label="diff prediction ("+str(np.round(np.sum(ts_temp1),4))+")")
	axarr[1].plot(ts_temp2,label="diff benchmark ("+str(np.round(np.sum(ts_temp2),4))+")")
	axarr[0].legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3,
				ncol=2, mode="expand", borderaxespad=0.)
	axarr[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
				ncol=2, mode="expand", borderaxespad=0.)
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

	plt.savefig('Output/pics/'+str(datetime.datetime.now())+'port_performance.png',bbox_inches='tight')
	plt.close()

