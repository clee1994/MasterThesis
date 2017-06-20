from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector, gen_xy, nearest, rnn_model
import logging, time

LOG_FILENAME = "./Output" + time.strftime("%Y%m%d_%H%M%S")
logging.getLogger().addHandler(logging.StreamHandler())

#load and preprocess data
[news_data, faulty_news] = load_news_data(False)
[prices, dates, names, lreturns, mu, sigma,dates_SP_weekly] = load_SP_data(False)

model = build_word2vec_model(False, news_data, faulty_news) 

[aggregated_news, dates_news] = get_news_vector(False,model, news_data, faulty_news)

#mean
[x,y] = gen_xy(aggregated_news,mu,dates_news,dates_SP_weekly)
mu_x_train = x[0:np.floor(np.shape(x)[0]*0.8),:]
mu_y_train = y[0:np.floor(np.shape(y)[0]*0.8),:]
mu_x_test = x[0:np.ceil(np.shape(x)[0]*0.2),:]
mu_y_test = y[0:np.ceil(np.shape(y)[0]*0.2),:]

#variance
[x,y] = gen_xy(aggregated_news,sigma,dates_news,dates_SP_weekly)
sigma_x_train = x[0:np.floor(np.shape(x)[0]*0.8),:]
sigma_y_train = y[0:np.floor(np.shape(y)[0]*0.8),:]
sigma_x_test = x[0:np.ceil(np.shape(x)[0]*0.2),:]
sigma_y_test = y[0:np.ceil(np.shape(y)[0]*0.2),:]

modelsRNNmu = list()
modelsRNNsigma = list()
loss_mu = list()
loss_sigma = list()
for i in range(np.shape(y_train)[1]):
	modelsRNNmu.append(rnn_model(mu_x_train,mu_y_train[:,1]))
	loss_mu.append(model.evaluate(mu_x_test, mu_y_test))
	modelsRNN.append(rnn_model(sigma_x_train,sigma_y_train[:,1]))
	loss_sigma.append(model.evaluate(sigma_x_test, sigma_y_test))




#learn news effect

#train

#test

#minimum variance portfolio projected CG

#sparsity minimum variance

#performance testing
