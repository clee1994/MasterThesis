

def load_news_data(alone, path):

	import numpy as np
	import pandas as pd
	import os as os
	import pickle
	import nltk.data
	from nltk.tokenize import RegexpTokenizer
	from progressbar import printProgressBar

	#load text data
	newsdates = os.listdir(path)
	news_data = []
	faulty_news = []
	sentences = []

	prog_st = 0
	l = len(newsdates) 
	printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

	for i in newsdates:
		prog_st += 1
		printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
		if not i.startswith('.'):
			newstitles = os.listdir(path + "/" + i)
			for j in newstitles:
				if not j.startswith('.'):
					try:
						file = open(path + "/" + i + "/" + j , "r") 
						text = file.read()
						file.close()
						[title, text] = text.split('\n',1)
						title = title[3:]
						[author, text] = text.split('\n',1)
						author = author[7:]
						[date, text] = text.split('\n',1)
						date = np.datetime64(pd.to_datetime(str(date[3:])) )
						[url, text] = text.split('\n',1)
						url = url[3:]
						
						try:
							[dummy, text] = text.split(') -',1)
						except ValueError:
							dummy = "no info"

						text = text.replace("\n", "")
						text = ' '.join(text.split())
						

						#sentences to list
						try:
							tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
						except:
							nltk.download()
							tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
						temp = tokenizer.tokenize(text)
						news_sub = []
						for j in temp:
							tokenizer = RegexpTokenizer(r'\w+')
							temp1 = tokenizer.tokenize(j)
							temp3 = []
							for k in temp1:
								if not k.isdigit():
									temp3.append(k)

							news_sub.append(temp3)

						#tokenization problemens U.S. -> 'U' 'S', company's -> 'company' 's'

						news_data.append([np.datetime64(i),j,title, author, \
							date, url, dummy, text, news_sub])

					except ValueError:
						faulty_news.append([i,j])

	news_data = sorted(news_data, key=lambda news: news[4] )

	if alone:
		f = open('./Data/processed_news_data', 'wb')
		pickle.dump([news_data, faulty_news], f)
		f.close()

	return [news_data, faulty_news]



def load_SP_data(alone):

	import numpy as np
	import pandas as pd
	import pickle
	import datetime

	#load SP500 Data
	raw_data = pd.read_csv('./Data/SP.csv', sep=',',header=None)
	temp = raw_data.values[1:,1:]
	prices = temp.astype(float)
	dates = np.array(raw_data.values[2:,0],dtype='datetime64')
	names = raw_data.values[0,1:-1]
	lreturns = np.diff(np.log(prices),n=1, axis=0)


	#weekly mean variance scaled..

	temp = dates[0].tolist()
	prev_date = temp.isocalendar()[1]
	temp_mean = np.array([lreturns[0,:]])
	cur_pos = 0
	mu_vec = np.zeros((1,505))
	sigma_vec = np.zeros((1,505))
	dates_SP_weekly = []
	dates_SP_weekly.append(dates[0])

	# for i in range(1,np.shape(lreturns)[0]):
	# 	temp = dates[i].tolist()
	# 	cur_date = temp.isocalendar()[1]
	# 	if cur_date == prev_date:
	# 		temp_mean = np.vstack((temp_mean, lreturns[i,:]))
	# 		prev_date = cur_date
	# 	else:
	# 		dates_SP_weekly.append(dates[i])
	# 		temp2 = np.mean(temp_mean, axis=0)
	# 		temp2 = np.reshape(temp2, (1,505))
	# 		temp3 = np.var(temp_mean, axis=0)
	# 		temp3 = np.reshape(temp2, (1,505))
	# 		mu_vec = np.vstack((mu_vec,temp2))
	# 		sigma_vec = np.vstack((sigma_vec,temp3))
	# 		cur_pos += 1
	# 		i += 1
	# 		temp_mean = np.array([lreturns[0,:]])
	# 		prev_date = cur_date

	if alone:
		f = open('./Data/processed_SP_data', 'wb')
		pickle.dump([prices, dates, names, lreturns, mu_vec, sigma_vec, dates_SP_weekly], f)
		f.close()


	return [prices, dates, names, lreturns, mu_vec, sigma_vec, dates_SP_weekly]
