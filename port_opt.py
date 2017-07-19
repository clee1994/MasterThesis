
def ret2prices(ret_series,base_value):
	import numpy as np

	prices = np.zeros([len(ret_series)+1])
	prices[0] = base_value
	for i in range(len(ret_series)):
		prices[i+1] = prices[i] * np.exp(ret_series[i])

	return prices





#cvxpy

def cv_opt(mu, Sigma, e_mu, glambda, h):
	from cvxpy import quad_form, Variable, sum_entries, Problem, Maximize, norm
	from sklearn.covariance import shrunk_covariance
	n = len(mu)
	w = Variable(n)
	#gamma = Parameter(sign='positive')
	ret = mu.T*w 
	try:
		risk = quad_form(w, Sigma)
	except:
		Sigma = shrunk_covariance(Sigma, shrinkage=0.1)
		risk = quad_form(w, Sigma)
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

	return w.value, ret.value, risk.value


