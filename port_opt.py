

def min_var_mu(mu, gamma, mu_p):

	import numpy as np


	lenm = len(mu)

	inv_sigma = np.linalg.pinv(gamma)
	one_v = np.ones([1,lenm])

	A = np.dot(one_v,np.dot(inv_sigma,mu))
	B = np.dot(one_v,np.dot(inv_sigma,np.transpose(one_v)))

	w_u = np.multiply(1/A,np.dot(inv_sigma,mu))
	w_u = np.transpose(np.reshape(w_u,[len(w_u),1]))
	w_1 = np.multiply(1/B,np.dot(inv_sigma,np.transpose(one_v)))

	D = np.subtract(np.dot(mu,np.transpose(w_u)), np.dot(mu, w_1)) 
	w = np.add(np.dot(np.divide(np.subtract(mu_p,np.dot(mu,w_1)),D),w_u), (np.dot(np.divide(np.subtract(np.dot(mu,np.transpose(w_u)),mu_p),D),np.transpose(w_1))))
	var_p = np.dot(np.dot(w,gamma),np.transpose(w))

	return w, var_p

def min_var(mu, gamma):
	import numpy as np

	lenm = len(mu)

	inv_sigma = np.linalg.pinv(gamma)
	one_v = np.ones([1,lenm])

	B = np.dot(one_v,np.dot(inv_sigma,np.transpose(one_v)))
	mu_p = np.dot(mu,np.dot(inv_sigma,np.transpose(one_v)))/B

	[w, var_p] = min_var_mu(mu,gamma,mu_p)

	return w, mu_p, var_p


def ret2prices(ret_series,base_value):
	import numpy as np

	prices = np.zeros([len(ret_series)+1])
	prices[0] = base_value
	for i in range(len(ret_series)):
		prices[i+1] = prices[i] * np.exp(ret_series[i])

	return prices



#testing for new optimization functions
import numpy as np

data_ts = np.random.rand(100,20)

mu = np.mean(data_ts,axis=0)
gamma = np.cov(np.transpose(data_ts))


mu_pp = list()
var_pp = list()
for i in (np.arange(100)/100):
	mu_pp.append(i)
	[w, var_p] = min_var_mu(mu,gamma,i)
	var_pp.append(var_p)

[w, mu_p, var_p] = min_var(mu, gamma)

import matplotlib.pyplot as plt
plt.figure()
plt.clf()
plt.plot(var_pp,mu_pp, 'ro')
plt.plot(var_p,mu_p, 'bo')
plt.show()





#cvxpy
def cv_opt(mu, Sigma, e_mu):
	from cvxpy import quad_form, Variable, sum_entries, Problem, Maximize
	n = len(mu)
	w = Variable(n)
	#gamma = Parameter(sign='positive')
	ret = mu.T*w 
	risk = quad_form(w, Sigma)
	if e_mu == None:
		prob = Problem(Maximize(ret - risk ), [sum_entries(w) == 1, w >= -1])
	else:
		prob = Problem(Maximize(ret - risk), [sum_entries(w) == 1, w >= -1, ret==e_mu])
	prob.solve() 

	return w.value, ret.value, risk.value

def cv_opt_l1(mu, Sigma, e_mu, glambda):
	from cvxpy import quad_form, Variable, sum_entries, Problem, Maximize
	n = len(mu)
	w = Variable(n)
	#gamma = Parameter(sign='positive')
	ret = mu.T*w 
	risk = quad_form(w, Sigma)
	if e_mu == None:
		prob = Problem(Maximize(ret - risk - glambda*norm(w,1) ), [sum_entries(w) == 1, w >= -1])
	else:
		prob = Problem(Maximize(ret - risk - glambda*norm(w,1)), [sum_entries(w) == 1, w >= -1, ret==e_mu])
	prob.solve() 

	return w.value, ret.value, risk.value


data_ts = np.random.rand(100,20)

mu = np.mean(data_ts,axis=0)
gamma = np.cov(np.transpose(data_ts))

mu_pp = list()
var_pp = list()
for z in np.linspace(0.4,0.5,50):
	mu_pp.append(float(z))
	[w, mu_p, var_p] = cv_opt(mu,gamma,float(z))
	var_pp.append(var_p)

[w, mu_p, var_p] = cv_opt(mu, gamma, None)

mu_pp1 = list()
var_pp1 = list()
for z in np.linspace(0.4,0.5,50):
	mu_pp1.append(float(z))
	[w1, mu_p, var_p] = cv_opt_l1(mu,gamma,float(z), 0.5)
	var_pp1.append(var_p)

[w1, mu_p1, var_p1] = cv_opt_l1(mu, gamma, None,0.5)

import matplotlib.pyplot as plt
plt.figure()
plt.clf()
plt.plot(np.array(var_pp).flatten(),mu_pp, 'ro')
plt.plot(np.diag(gamma), mu, 'bo')
plt.plot(var_p, mu_p, 'go')
plt.plot(np.array(var_pp1).flatten(),mu_pp1, 'rx')
plt.plot(var_p1, mu_p1, 'gx')
plt.show()

plt.figure()
plt.bar(w)
plt.show()

plt.figure()
plt.bar(w1)
plt.show()
















#cvxopt
data_ts = np.random.rand(100,4)

mu = np.mean(data_ts,axis=0)
gamma = np.cov(np.transpose(data_ts))

mu_pp = list()
var_pp = list()
for z in np.linspace(0.1,20,10):
	mu_pp.append(float(z))
	[w, mu_p, var_p] = opt_min(mu,gamma,float(z))
	var_pp.append(var_p)

[w, mu_p, var_p] = opt_min(mu, gamma, None)

import matplotlib.pyplot as plt
plt.figure()
plt.clf()
plt.plot(np.array(var_pp).flatten(),mu_pp, 'ro')
plt.plot(np.diag(gamma), mu, 'bo')
plt.plot(var_p, mu_p, 'go')
plt.show()

def opt_min(mu, gamma, e_mu):

	from cvxopt import solvers
	from cvxopt import matrix as opt_matrix
	import numpy as np
	solvers.options['show_progress'] = False
	#solvers.options['maxiters'] = 5000
	#solvers.options['abstol'] = 1e-15
	#solvers.options['reltol'] = 1e-15
	#solvers.options['feastol'] = 1e-15
	#solvers.options['refinement'] = 10


	mu = opt_matrix(mu)
	gamma = opt_matrix(gamma)


	n = len(mu)

	G = opt_matrix(0.0, (n,n))
	G[::n+1] = -1.0
	h = opt_matrix(0.0, (n,1))
	h[:] = 0.0
	A = opt_matrix(1.0, (1,n))
	b = opt_matrix(1.0)


	#emu = 0.001

	if e_mu != None:
		xs = np.array(solvers.qp(e_mu*gamma, -mu, G, h, A, b, 'mosek')['x'])
	else:
		xs = np.array(solvers.qp(gamma, -mu, G, h, A, b, 'mosek')['x'])
	mu_p = np.dot(np.transpose(xs),mu)
	var_p = np.dot(np.dot(np.transpose(xs), gamma), (xs))

	return xs, mu_p, var_p





#split bregman -> problem

#definiton
tol = 0.0001

#code
data_ts = np.random.rand(100,20)
data_ts = np.transpose(data_ts)

[lambdap,rho] = bregman_parameter(data_ts, tol)
tol = 0.0000001
[w, mu_p, var_p] = bregman(rho, lambdap, tol, data_ts)

import matplotlib.pyplot as plt
plt.figure()
plt.clf()
plt.plot(var_pp,mu_pp, 'ro')
plt.plot(var_p,mu_p, 'bo')
plt.show()



def bregman_parameter(data_ts,tol):
	lambdap_range = np.linspace(0,7,25)
	rho_range = np.linspace(0.1,17.5,25)

	var_p = list()

	for i in lambdap_range:
		[wt, mu_pt, var_pt] = bregman(8, i, tol, data_ts)
		var_p.append(var_pt)

	lambdap_f = lambdap_range[var_p.index(min(var_p))]
	var_p = list()

	for i in rho_range:
		[w, mu_p, var_pt] = bregman(i, lambdap_f, tol, data_ts)
		var_p.append(var_pt)
	rho_f = rho_range[var_p.index(min(var_p))]

	return lambdap_f, rho_f




#faulty bregman
def shrinkage_operator(x,gamma):
	return (x/np.abs(x)) * np.max([np.abs(x)-gamma,0])

def bregman(rho, lambdap, tol,data_ts):
	mu = np.mean(data_ts,axis=1)
	gamma = np.cov(np.transpose(data_ts),rowvar=False)
	len_s = len(mu)

	K = 1000

	#beta uncertainty -> for mean
	bootstat = np.zeros([K,np.shape(data_ts)[0]])
	for i in range(K):
		bootstat[i,:] = np.mean(data_ts[:,np.random.randint(0,np.shape(data_ts)[1]-1, int(np.floor(np.shape(data_ts)[1]*0.9)))])

	mu_err = np.abs(np.subtract(bootstat, np.reshape(np.mean(bootstat,axis=0),[1,len(data_ts)])))
	beta = np.mean(mu_err, axis=0)


	#alpha uncertainty -> for cov
	bootstat = np.zeros([K,np.shape(data_ts)[0],np.shape(data_ts)[0]])
	for i in range(K):
		bootstat[i,:,:] = np.cov(data_ts[:,np.random.randint(0,np.shape(data_ts)[1]-1, int(np.floor(np.shape(data_ts)[1]*0.9)))])
	cov_err = np.abs(np.subtract(bootstat, np.mean(bootstat,axis=0)))
	alpha = np.mean(cov_err, axis=0)


	B = np.multiply(np.identity(len(beta)),beta)

	b = list()
	b.append([])
	b.append(np.zeros([len_s,1]))

	d = list()
	d.append([])
	d.append(np.zeros([len_s,1]))

	w = list()
	w.append(-np.ones([len_s,1]))
	w.append(np.zeros([len_s,1]))

	R = np.add(np.multiply(rho, gamma) , alpha)


	while np.linalg.norm(np.subtract(w[-1], w[-2])) > tol:
		w.append( np.linalg.solve( np.add(R , np.dot(np.transpose(B), B )),  ( np.add(np.reshape(mu,[len_s,1]), np.multiply(lambdap, np.dot(np.transpose(B) ,np.subtract(d[-1],b[-1]) ))) )) )
		d.append(np.zeros([len_s,1]))
		b.append(np.zeros([len_s,1]))
		for i in range(len_s):
			d[-1][i] = shrinkage_operator((beta[i]*w[-1][i] +b[-2][i]), (1/lambdap))
			b[-1][i] = b[-2][i] + beta[i] * w[-1][i] - d[-1][i]

	mu_p = np.dot(np.transpose(w[-1]),mu)
	var_p = np.dot(np.dot(np.transpose(w[-1]), gamma), (w[-1]))

	return w[-1], mu_p, var_p

