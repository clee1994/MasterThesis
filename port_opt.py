
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go


# def min_var( exp_ret, mu, sigma ):
#     S_inv = np.linalg.inv(sigma)
#     pi_mu = np.divide(np.dot(S_inv,exp_ret), np.sum(np.dot(S_inv,exp_ret),axis=0))
#     pi_1 = np.divide(np.sum(S_inv,axis=0), np.transpose(np.sum(S_inv)))
#     lambda_demoninator = np.dot(np.transpose(exp_ret),pi_mu) - np.dot(np.transpose(exp_ret),pi_1)
#     ll = np.divide((mu - np.dot(np.transpose(exp_ret),pi_1)), lambda_demoninator) 
#     return (pi_mu * ll + pi_1 * (1-ll))


# def min_var(Sigma, mean_returns, mu_p):
#     pi_mu = np.linalg.solve(Sigma,mean_returns)
#     pi_mu/= pi_mu.sum()
#     pi_l = np.linalg.solve(Sigma,np.ones_like(mean_returns))
#     pi_l/=pi_l.sum()
#     lambda_demoninator = mean_returns.dot(pi_mu) - mean_returns.dot(pi_l)
#     ll = np.array((mu_p - mean_returns.dot(pi_l))/lambda_demoninator)
#     # previous line: to convert into array in case that mu_p is a single number
#     ll.shape=(ll.size,1)
#     return pi_mu * ll + pi_l * (1-ll)


# numerator = np.dot(mu,(np.dot(np.linalg.inv(gamma),np.ones([lenm,1]))))
# denominator =  np.dot(np.ones([1,lenm]),(np.dot(np.linalg.inv(gamma),np.ones([lenm,1]))))
# mu_p = numerator/denominator;

# w = min_var(gamma, np.transpose(mu), mu_p)
# var_p = np.dot(np.dot(w,gamma),np.transpose(w))

data_ts = np.random.rand(100,20)


mu = np.mean(data_ts,axis=0)
gamma = np.cov(np.transpose(data_ts))

lenm = len(mu)

inv_sigma = np.linalg.inv(gamma)

A = np.dot(np.ones([1,lenm]),np.dot(inv_sigma,mu))
B = np.dot(np.ones([1,lenm]),np.dot(inv_sigma,np.ones([lenm,1])))


w_u = np.multiply(1/A,np.dot(inv_sigma,mu))
w_1 = np.multiply(1/B,np.dot(inv_sigma,np.ones([lenm,1])))
mu_p = A/B
D = np.subtract(np.dot(mu,w_u), np.dot(mu, w_1)) 
w = np.add(np.multiply(np.divide(np.subtract(mu_p,np.dot(mu,w_1)),D),w_u), np.transpose(np.multiply(np.divide(np.subtract(np.dot(mu,w_u),mu_p),D),w_1)))
var_p = np.dot(np.dot(w,gamma),np.transpose(w))



mu = np.random.rand(30)
gamma = np.random.rand(30,30)
lenm = len(mu)

inv_sigma = np.linalg.inv(gamma)
A = np.dot(np.ones([1,lenm]),np.dot(inv_sigma,mu))
B = np.dot(np.ones([1,lenm]),np.dot(inv_sigma,np.ones([lenm,1])))
w_u = np.multiply(1/A,np.dot(inv_sigma,mu))
w_1 = np.multiply(1/B,np.dot(inv_sigma,np.ones([lenm,1])))

var = np.zeros([100,1])
for i in range(100):
	cur_mu = 1/100
	print(i/100)
	w = np.add(np.multiply(np.divide(np.subtract(cur_mu,np.dot(mu,w_1)),D),w_u), np.transpose(np.multiply(np.divide(np.subtract(np.dot(mu,w_u),cur_mu),D),w_1)))
	var[i] = np.dot(np.dot(w,gamma),np.transpose(w))



# Create a trace
trace = go.Scatter(
    y = np.arange(1,101)/100,
    x = var,
    mode = 'markers'
)

trace1 = go.Scatter(
	x = mu,
    y = np.diagonal(gamma),
    mode = 'markers'
)

data_plot = [trace,trace1]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')



# projected CG
import numpy

def nullspace(A, atol=1e-13, rtol=0):
	#http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
	from numpy.linalg import svd

	A = np.atleast_2d(A)
	u, s, vh = svd(A)
	tol = max(atol, rtol * s[0])
	nnz = (s >= tol).sum()
	ns = vh[nnz:].conj().T
	return ns


mu = np.random.rand(30)
gamma = np.random.rand(30,30)
lenm = len(mu)

A = np.ones([1,len(gamma)])
b = 1
c = np.zeros([len(gamma),1])


Z = nullspace(gamma)


x = A'*((A*A')\b);
r = gamma*x +c; 
%P = Z*((Z'*eye(size(Z,1))*Z)\Z'); %doubts
P = Z*((Z'*eye(size(Z,1))*Z)\Z');
g = P*r;
d = -g;


while (r'*g) > tol %(rz'*(Wzz\rz)) < tol 
    alpha = (r' * g) / (d'*gamma*d);
    x = x + alpha*d;
    rp = r + alpha * gamma * d;
    gp = P * rp;
    beta = (rp' * gp)/(r'*g);
    d = -gp + beta*d;
    g = gp;
    r = rp;
end

time = toc;
w = x;

mu_p =w'*mu';
var_p =w'*gamma*w;
