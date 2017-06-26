from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector
from learning import  gen_xy, nearest, rnn_model
from port_opt import min_var_mu, min_var, ret2prices
import numpy as np
import gensim
import pickle
import os as os

import sys
import datetime
import gc

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
#sys.stdout = open('Output/logfile'+str(datetime.datetime.now()), 'w')


#modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
n_forward_list=[1,3,7]
n_past_list = [20,60,100]
test_split = 0.15
validation_split = 0.12
batch_size = 40
epoches_list = [1,3]
word_min_count_list = [10,160,400]
feature_number_list = [150, 200, 250, 300, 350]
firms_used = 30


#def main_comp(path_to_news_files,n_forward,n_past,test_split,validation_split, batch_size,epoches,word_min_count, feature_number, firms_used):
# for feature_number in feature_number_list:
#load and preprocess data
#print('Start reading in news:')
#[news_data, faulty_news] = load_news_data(False,path_to_news_files)
#print('Successfully read all news')
print('--------------------------------------')
print('Start reading in SP500 data:')
[prices, dates, names, lreturns, mu, sigma,dates_SP_weekly] = load_SP_data(False)
print('Successfully read all data')

# #train word2vec model
# for word_min_count in word_min_count_list:
# 	for feature_number in feature_number_list:
# 		print('--------------------------------------')
# 		print('Start building word2vec model:')
# 		model = build_word2vec_model(False,feature_number,word_min_count, news_data, faulty_news)
# 		model.save("Output/models/"+str(word_min_count)+"_"+str(feature_number))
		
# 		print('Successfully build model')
# 		print('--------------------------------------')

# 		#transform news to vectors
# 		print('Start aggregating news data:')
# 		[aggregated_news, dates_news] = get_news_vector(False,model,feature_number, news_data, faulty_news)
# 		print('Successfully aggregated data')
# 		print('--------------------------------------')

# 		pickle.dump( [aggregated_news,dates_news], open( "Output/aggr_news/"+str(word_min_count)+"_"+str(feature_number), "wb" ) )

# 		del model, aggregated_news, dates_news


model_list = os.listdir("Output/aggr_news")

for n_forward in n_forward_list:
	for n_past in n_past_list:
		for cur_m in model_list:

			model = gensim.models.Word2Vec.load("Output/models/"+cur_m)
			[aggregated_news,dates_news] = pickle.load( open( "Output/aggr_news/"+cur_m, "rb" ) )

			#generat x and y data/train/test
			print('Start generating mu train and test data:')
			[x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test] = gen_xy(
				aggregated_news,
				lreturns[:,0:firms_used],
				dates,
				dates_news,
				n_forward,
				n_past,
				True,
				names,
				test_split)
			print('Successfully generated mu train and test data')
			print('--------------------------------------')

			pickle.dump( [x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test], open( "Output/xy/"+str(n_forward)+"_"+str(n_past), "wb" ) )

			firms_used = np.shape(y_test)[1]

			for epoches in epoches_list:
				#modelsRNNmu = list()
				#modelsRNNsigma = list()
				loss_mu = list()
				#loss_sigma = list()
				predict_y = list()

				# modelsRNNmu.append(rnn_model(x_train,y_train[:,4],validation_split,batch_size,epoches))
				# temp1 = modelsRNNmu[0].evaluate(x_test, y_test[:,4])
				# loss_mu.append(temp1)
				for i in range(0,firms_used):
					print('Start building prediction model:')
					tempRNN = rnn_model(x_train,y_train[:,i],validation_split,batch_size,epoches)
					tempRNN.save("Output/modelsKeras/"+cur_m+"_"+str(n_forward)+"_"+str(n_past)+"_"+str(epoches))
					#modelsRNNmu.append(tempRNN)
					
					loss_text = tempRNN.evaluate(x_test, y_test[:,i])
					print(loss_text)
					#loss_mu.append(temp1)
					#print(temp1)
					# modelsRNNsigma.append(rnn_model(sigma_x_train,sigma_y_train[:,i]))
					# temp1 = modelsRNNsigma[i].evaluate(sigma_x_test, sigma_y_test[:,i])
					# loss_sigma.append(temp1)
					# print(temp1)
					print('Successfully generated model')
					print('--------------------------------------')
					temp_y_pred = tempRNN.predict(x_test, batch_size=128)
					
					
					plt.figure()
					plt.clf()
					plt.plot(y_test[:,i],label= "true y")
					plt.plot(np.squeeze(temp_y_pred),label="predicted y")
					plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
				           ncol=2, mode="expand", borderaxespad=0.)
					plt.text(0.95, 0.95, str(loss_text))
					plt.savefig('Output/pics/'+str(datetime.datetime.now())+"_"+str(i)+"_"+cur_m+"_"+str(n_forward)+"_"+str(n_past)+"_"+str(epoches)+'pred_true.png')
					plt.close()
					plt.close("all")
					#del predict_y
					del tempRNN, loss_text
					predict_y.append(temp_y_pred)
					del temp_y_pred
					gc.collect()
					
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

					mu = np.nanmean(lreturns[(ind_d-n_past):ind_d, firm_ind],axis=0)

					improved_mu = np.zeros(firms_used)
					#using news to improve
					for j in range(firms_used):
						#absolutly ridiculus change
						#temp_data_input_predict = np.reshape(x_test[i,:], (1,) + x_test[i,:].shape)
						mu_change = predict_y[j][i]
						#[j].predict(temp_data_input_predict)
						improved_mu[j] = mu[j] + mu_change

					gamma = np.cov(lreturns[(ind_d-n_past):ind_d, firm_ind],rowvar=False)

					[w, mu_p, var_p] = min_var(mu, gamma)
					[i_w, i_mu_p, i_var_p] = min_var(improved_mu, gamma)
					realized_mu.append(np.dot(w,lreturns[ind_d,firm_ind]))
					i_realized_mu.append(np.dot(i_w,lreturns[ind_d,firm_ind]))


				#visualization of results!
				realized_mu = np.array(realized_mu).flatten()
				value_over_time = ret2prices(realized_mu,100)

				i_realized_mu = np.array(i_realized_mu).flatten()
				i_value_over_time = ret2prices(i_realized_mu,100)


				import matplotlib.pyplot as plt
				plt.figure() 
				plt.clf()
				plt.plot(value_over_time , 'r', label='ordinary min var')
				plt.plot(i_value_over_time , 'b', label='improved min var portfolio')

				#plt.plot(np.subtract(value_over_time,i_value_over_time), 'g', label='improved min var portfolio')
				plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
				           ncol=2, mode="expand", borderaxespad=0.)

				plt.savefig('Output/pics/'+str(datetime.datetime.now())+cur_m+"_"+str(n_forward)+"_"+str(n_past)+"_"+str(epoches)+'port_performance.png')
				plt.close()
				del realized_mu, i_realized_mu, value_over_time, i_value_over_time, predict_y
				gc.collect()

			del aggregated_news,dates_news, model
			del x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test	
			gc.collect()
		try:
			del aggregated_news,dates_news, model
			gc.collect()
		try:
			del x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test	
			gc.collect()
		try:
			del realized_mu, i_realized_mu, value_over_time, i_value_over_time, predict_y
			gc.collect()
	try:
		del aggregated_news,dates_news, model
		gc.collect()
	try:
		del x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test	
		gc.collect()
	try:
		del realized_mu, i_realized_mu, value_over_time, i_value_over_time, predict_y
		gc.collect()
try:
	del aggregated_news,dates_news, model
	gc.collect()
try:
	del x_train,y_train,x_test,y_test,used_stocks,x_dates_train, x_dates_test	
	gc.collect()
try:
	del realized_mu, i_realized_mu, value_over_time, i_value_over_time, predict_y
	gc.collect()









