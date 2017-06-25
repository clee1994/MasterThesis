from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector
from learning import  gen_xy, nearest, rnn_model
from port_opt import min_var_mu, min_var, ret2prices
import numpy as np


#modifiable variables
path_to_news_files = "./Data/ReutersNews106521"
n_forward_list=[1,3,7]
n_past_list = [20,60,100]
test_split = 0.15
validation_split = 0.12
batch_size = 40
epoches_list = [1,2,3]
word_min_count_list = [10,160,400]
feature_number_list = [150, 200, 250, 300, 350]
firms_used = 5


#def main_comp(path_to_news_files,n_forward,n_past,test_split,validation_split, batch_size,epoches,word_min_count, feature_number, firms_used):
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
for word_min_count in word_min_count_list:
	for feature_number in feature_number_list:
		print('--------------------------------------')
		print('Start building word2vec model:')
		model = build_word2vec_model(False,feature_number,word_min_count, news_data, faulty_news)
		model.save("Output/models/"+str(word_min_count)+"_"+str(feature_number))
		del model
		print('Successfully build model')
		print('--------------------------------------')

