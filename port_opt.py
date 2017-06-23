

def min_var_mu(mu, gamma, mu_p):

	import numpy as np


	lenm = len(mu)

	inv_sigma = np.linalg.inv(gamma)
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

	inv_sigma = np.linalg.inv(gamma)
	one_v = np.ones([1,lenm])

	B = np.dot(one_v,np.dot(inv_sigma,np.transpose(one_v)))
	mu_p = np.dot(mu,np.dot(inv_sigma,np.transpose(one_v)))/B

	[w, var_p] = min_var_mu(mu,gamma,mu_p)

	return w, mu_p, var_p




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
