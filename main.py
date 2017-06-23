from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector
from learning import  gen_xy, nearest, rnn_model
import numpy as np

#modifiable variables
path_to_news_files = "./Data_small/ReutersNews106521"
n_forward=5
n_past = 30
test_split = 0.1
validation_split = 0.1
batch_size = 32
epoches = 5
word_min_count = 10
feature_number = 300


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
[aggregated_news, dates_news] = get_news_vector(False,model, news_data, faulty_news)
print('Successfully aggregated data')
print('--------------------------------------')


#generat x and y data/train/test
print('Start generating mu train and test data:')
[x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test] = gen_xy(
	aggregated_news,
	lreturns[:,1:10],
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
for i in range(0,10):
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


import datetime
no_t = 3
predict_y = modelsRNNmu[no_t].predict(x_test, batch_size=128)
true_y_com = np.zeros(len(predict_y))
pred_y_com = np.zeros(len(predict_y))

temp = x_dates_test[1].tolist()
prev_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
j = 0
k = 0
true_y_com[0] = y_test[0,no_t]

for i in range(len(predict_y)):
	temp = x_dates_test[i].tolist()
	cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))
	
	if prev_d == cur_d:
		prev_d = cur_d
		#true_y_com[j,k] += y_test[i,0]
		pred_y_com[j] += predict_y[i]
		k += 1
	else:
		j += 1
		true_y_com[j] = y_test[i,no_t]
		pred_y_com[j] += predict_y[i]
		k = 0
		prev_d = cur_d


# import matplotlib.pyplot as plt

# plt.figure(1)

# plt.plot(true_y_com[0:j+1], 'r', pred_y_com[0:j+1], 'bs')
# plt.show()


# #plt.subplot(212)
# #plt.plot(lreturns[:,0])



# predict_y = modelsRNNmu[0].predict(x_test, batch_size=128)

# plt.subplot(211)
# plt.plot(predict_y[:,0])

# plt.subplot(212)
# plt.plot(y_test[:,0])
# plt.show()



#visualize results


#learn news effect

#train

#test

#minimum variance portfolio projected CG

#sparsity minimum variance

#performance testing
