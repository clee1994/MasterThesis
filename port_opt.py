

def ret2prices(ret_series,base_value):
	import numpy as np

	prices = np.zeros([len(ret_series)+1])
	prices[0] = base_value
	for i in range(len(ret_series)):
		#probably cheating but... we pray that there wont be any nan
		if np.isnan(ret_series[i]):
			prices[i+1] = prices[i]
		prices[i+1] = prices[i] * np.exp(ret_series[i])

	return prices

def is_pos_def(x):
	import numpy as np
	return np.all(np.linalg.eigvals(x) > 0)

#cvxpy
def cv_opt(mu, Sigma, e_mu, glambda, h):
	from cvxpy import quad_form, Variable, sum_entries, Problem, Maximize, norm
	from sklearn.covariance import shrunk_covariance
	import numpy as np
	n = len(mu)
	w = Variable(n)
	#gamma = Parameter(sign='positive')
	ret = mu.T*w 


	if is_pos_def(Sigma):
		risk = quad_form(w, Sigma)
	else:
		for i in np.linspace(0.01,10000,1000000):
			test = shrunk_covariance(Sigma, shrinkage=i)
			if is_pos_def(test):
				Sigma = test
				break
		risk = quad_form(w, Sigma)
		if not is_pos_def(Sigma):
			print('Here you got a serious problem', flush=True)
			print(Sigma)
			rw = np.empty((len(mu)))
			rw[:] = np.nan
			rr = np.nan
			rri = np.nan
			return rw, rr, rri

	if glambda == None:
		if e_mu == None:
			prob = Problem(Maximize(ret - risk), [sum_entries(w) == 1, w >= h])
		else:
			prob = Problem(Maximize(ret - risk), [sum_entries(w) == 1, w >= h, ret==e_mu])
	else:
		if e_mu == None:
			prob = Problem(Maximize(ret - risk - glambda*norm(w,1) ), [sum_entries(w) == 1, w >= h])
		else:
			prob = Problem(Maximize(ret - risk - glambda*norm(w,1)), [sum_entries(w) == 1, w >= h, ret==e_mu])
	prob.solve() 


	try:
		rw = w.value
		rr = ret.value
		rri = risk.value
	except:
		rw = np.empty((len(mu)))
		rw[:] = np.nan
		rr = np.nan
		rri = np.nan

	return rw, rr, rri


