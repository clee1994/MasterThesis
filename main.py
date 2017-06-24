from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector
from learning import  gen_xy, nearest, rnn_model
from port_opt import min_var_mu, min_var, ret2prices
import numpy as np

#modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
n_forward=4
n_past = 60
test_split = 0.15
validation_split = 0.12
batch_size = 40
epoches = 5
word_min_count = 160
feature_number_list = [150, 200, 220, 250, 300, 350]
feature_number = 210

firms_used = 30

# for feature_number in feature_number_list:
#load and preprocess data
print('Start reading in news:')
[news_data, faulty_news] = load_news_data(False,path_to_news_files)
print('Successfully read all news')
print('--------------------------------------')
print('Start reading in SP500 data:')
[prices, dates, names, lreturns, mu, sigma,dates_SP_weekly] = load_SP_data(False)
print('Successfully read all data')


#train word2vec model
print('--------------------------------------')
print('Start building word2vec model:')
model = build_word2vec_model(False,feature_number,word_min_count, news_data, faulty_news) 
print('Successfully build model')
print('--------------------------------------')

#transform news to vectors
print('Start aggregating news data:')
[aggregated_news, dates_news] = get_news_vector(False,model,feature_number, news_data, faulty_news)
print('Successfully aggregated data')
print('--------------------------------------')


#generat x and y data/train/test
print('Start generating mu train and test data:')
[x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test] = gen_xy(
	aggregated_news,
	lreturns[:,1:20],
	dates,
	dates_news,
	n_forward,
	n_past,
	True,
	names,
	test_split)
print('Successfully generated mu train and test data')
print('--------------------------------------')


modelsRNNmu = list()
modelsRNNsigma = list()
loss_mu = list()
loss_sigma = list()

# modelsRNNmu.append(rnn_model(x_train,y_train[:,4],validation_split,batch_size,epoches))
# temp1 = modelsRNNmu[0].evaluate(x_test, y_test[:,4])
# loss_mu.append(temp1)
for i in range(0,firms_used):
	print('Start building prediction model:')
	modelsRNNmu.append(rnn_model(x_train,y_train[:,i],validation_split,batch_size,epoches))
	temp1 = modelsRNNmu[i].evaluate(x_test, y_test[:,i])
	loss_mu.append(temp1)
	print(temp1)
	# modelsRNNsigma.append(rnn_model(sigma_x_train,sigma_y_train[:,i]))
	# temp1 = modelsRNNsigma[i].evaluate(sigma_x_test, sigma_y_test[:,i])
	# loss_sigma.append(temp1)
	# print(temp1)
	print('Successfully generated model')
	print('--------------------------------------')




#plotting results so far
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime

for i in range(len(modelsRNNmu)):
	predict_y = modelsRNNmu[i].predict(x_test, batch_size=128)
	plt.figure()
	plt.clf()
	plt.plot(y_test[:,i],label= "true y")
	plt.plot(np.squeeze(predict_y),label="predicted y")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

	plt.savefig('Output/pics/'+str(datetime.datetime.now())+str(i)+'pred_true.png')
	plt.close()
#del news_data, faulty_news, model, aggregated_news, dates_news, x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test, modelsRNNmu


#backtesting portfolio opt with improved estimates


#firm indices
firm_ind = list()

for i in range(firms_used):
	firm_ind.append(list(names).index(used_stocks[i]))

realized_mu = list()
i_realized_mu = list()

for i in range(len(x_dates_test)):
	temp = x_dates_test[i].tolist()
	cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
	try:
		ind_d = list(dates).index(cur_d)
	except:
		print('cant happen')

	mu = np.mean(lreturns[(ind_d-n_past):ind_d, firm_ind],axis=0)

	improved_mu = np.zeros(firms_used)
	#using news to improve
	for j in range(firms_used):
		#absolutly ridiculus change
		temp_data_input_predict = np.reshape(x_test[i,:], (1,) + x_test[i,:].shape)
		mu_change = modelsRNNmu[j].predict(temp_data_input_predict)
		improved_mu[j] = mu[j] + mu_change

	gamma = np.cov(lreturns[(ind_d-n_past):ind_d, firm_ind],rowvar=False)

	[w, mu_p, var_p] = min_var(mu, gamma)
	[i_w, i_mu_p, i_var_p] = min_var(improved_mu, gamma)
	realized_mu.append(np.dot(w,lreturns[ind_d,firm_ind]))
	i_realized_mu.append(np.dot(i_w,lreturns[ind_d,firm_ind]))


#visualization of results!
realized_mu = np.squeeze(realized_mu)
value_over_time = ret2prices(realized_mu,100)

i_realized_mu = np.squeeze(i_realized_mu)
i_value_over_time = ret2prices(i_realized_mu,100)


import matplotlib.pyplot as plt
plt.figure() 
plt.clf()
plt.plot(value_over_time , 'r', label='ordinary min var')
plt.plot(i_value_over_time , 'b', label='improved min var portfolio')
plt.plot(np.subtract(value_over_time,i_value_over_time), 'g', label='improved min var portfolio')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.savefig('Output/pics/'+str(datetime.datetime.now())+str(i)+'port_performance.png')
plt.close()



