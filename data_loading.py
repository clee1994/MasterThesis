

def load_news_data(path, stop_word_removal):

	import numpy as np
	import pandas as pd
	import os as os
	import nltk.data
	from nltk.tokenize import RegexpTokenizer
	from progressbar import printProgressBar
	from nltk.corpus import stopwords #stopwords removal

	#load text data
	newsdates = os.listdir(path)
	news_data = []
	faulty_news = []
	sentences = []
	stops = set(stopwords.words('english'))#stopwords removal

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
									if stop_word_removal:
										if k.lower() not in stops: #stopword removal
											temp3.append(k)
									else:
										temp3.append(k)

							news_sub.append(temp3)

						#tokenization problemens U.S. -> 'U' 'S', company's -> 'company' 's'

						news_data.append([np.datetime64(i),j,title, author, \
							date, url, dummy, text, news_sub])

					except ValueError:
						faulty_news.append([i,j])

	news_data = sorted(news_data, key=lambda news: news[4] )


	return [news_data, faulty_news]



def load_SP_data(stock_names,pref_number):

	import numpy as np
	import pandas as pd
	import pickle
	import datetime

	#load SP500 Data
	raw_data = pd.read_csv('./Data/SP.csv', sep=',',header=None)
	temp = raw_data.values[1:,1:]
	prices = temp.astype(float)
	dates = np.array(raw_data.values[2:,0],dtype='datetime64')
	names = raw_data.values[0,1:]
	lreturns = np.diff(np.log(prices),n=1, axis=0)

	#remove not listed stocks
	ind_stocks = list()
	for i in stock_names:
		temp = list(names).index(i)
		if (np.sum(np.isnan(lreturns[:,temp]))/np.shape(lreturns)[0]) < 0.05:
			ind_stocks.append(temp)
		del temp

	ind_stocks = np.array(ind_stocks)
	if len(ind_stocks) > pref_number:
		ind_stocks = ind_stocks[:pref_number]

	lreturns = lreturns[:,ind_stocks]
	prices = prices[:,ind_stocks]



	return [prices, dates, names, lreturns]
