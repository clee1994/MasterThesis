from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector
from learning import  gen_xy, nearest, rnn_model
import numpy as np

#modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
n_forward=13
n_past = 60
test_split = 0.15
validation_split = 0.12
batch_size = 40
epoches = 5
word_min_count = 160
feature_number = 200


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
for i in range(0,5):
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

for i in range(len(modelsRNNmu)):
	predict_y = modelsRNNmu[i].predict(x_test, batch_size=128)
	plt.figure()
	plt.clf()
	plt.plot(y_test[:,i],label= "true y")
	plt.plot(np.squeeze(predict_y),label="predicted y")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

	plt.savefig('Output/pics/'+str(i)+'pred_true.png')


#save stuf
import pickle

pickle.dump((x_train, x_test, y_train, y_test) , open( "Output/xy_data", "wb" ) )
model.save("Output/word2vec_model")
for i in range(len(modelsRNNmu)):
	modelsRNNmu[i].save("Output/keras_model" + str(i))



