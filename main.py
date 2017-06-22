from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector, gen_xy, nearest, rnn_model
import numpy as np

#load and preprocess data
print('Start reading in news:')
[news_data, faulty_news] = load_news_data(False)
print('Successfully read all news')
print('--------------------------------------')
print('Start reading in SP500 data:')
[prices, dates, names, lreturns, mu, sigma,dates_SP_weekly] = load_SP_data(False)
print('Successfully read all data')
print('--------------------------------------')
print('Start building word2vec model:')
model = build_word2vec_model(False, news_data, faulty_news) 
print('Successfully build model')
print('--------------------------------------')

print('Start aggregating news data:')
[aggregated_news, dates_news] = get_news_vector(False,model, news_data, faulty_news)
print('Successfully aggregated data')
print('--------------------------------------')


#mean
print('Start generating mu train and test data:')
[x,y] = gen_xy(aggregated_news,mu,dates_news,dates_SP_weekly)
mu_x_train = x[0:np.floor(np.shape(x)[0]*0.8),:]
mu_y_train = y[0:np.floor(np.shape(y)[0]*0.8),:]
mu_x_test = x[0:np.ceil(np.shape(x)[0]*0.2),:]
mu_x_test = np.reshape(mu_x_test, mu_x_test.shape + (1,))
mu_y_test = y[0:np.ceil(np.shape(y)[0]*0.2),:]
print('Successfully generated mu train and test data')
print('--------------------------------------')


#variance
print('Start generating sigma train and test data:')
[x,y] = gen_xy(aggregated_news,sigma,dates_news,dates_SP_weekly)
sigma_x_train = x[0:np.floor(np.shape(x)[0]*0.9),:]
sigma_y_train = y[0:np.floor(np.shape(y)[0]*0.9),:]
sigma_x_test = x[0:np.ceil(np.shape(x)[0]*0.1),:]
sigma_x_test = np.reshape(sigma_x_test, sigma_x_test.shape + (1,))
sigma_y_test = y[0:np.ceil(np.shape(y)[0]*0.1),:]
print('Successfully generated sigma train and test data')
print('--------------------------------------')

used_stocks = list()
bad_stocks = list()

#drop stocks with insufficient data
for i in range(np.shape(mu_y_train)[1]):
	if (np.sum(np.isnan(mu_y_train[:,i]))/len(mu_y_train[:,i])) > 0.1:
		bad_stocks.append(i)
	else:
		used_stocks.append(names[i])

mu_y_train = np.delete(mu_y_train,bad_stocks,1)
sigma_y_train = np.delete(sigma_y_train,bad_stocks,1)

modelsRNNmu = list()
modelsRNNsigma = list()
loss_mu = list()
loss_sigma = list()
for i in range(5):
	print('Start building prediction model:')
	modelsRNNmu.append(rnn_model(mu_x_train,mu_y_train[:,i]))
	temp1 = modelsRNNmu[i].evaluate(mu_x_test, mu_y_test[:,i])
	loss_mu.append(temp1)
	print(temp1)
	modelsRNNsigma.append(rnn_model(sigma_x_train,sigma_y_train[:,i]))
	temp1 = modelsRNNsigma[i].evaluate(sigma_x_test, sigma_y_test[:,i])
	loss_sigma.append(temp1)
	print(temp1)
	print('Successfully generated model')
	print('--------------------------------------')



#visualize results


#learn news effect

#train

#test

#minimum variance portfolio projected CG

#sparsity minimum variance

#performance testing
