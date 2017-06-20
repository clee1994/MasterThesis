

def load_data(alone, n):

	import numpy as np
	import pandas as pd
	import os as os
	import pickle
	import nltk.data
	from nltk.tokenize import RegexpTokenizer

	#load SP500 Data
	raw_data = pd.read_csv('./Data/SP.csv', sep=',',header=None)
	temp = raw_data.values[1:,1:]
	prices = temp.astype(float)
	dates = np.array(raw_data.values[1:-1,0],dtype='datetime64')
	names = raw_data.values[0,1:-1]
	lreturns = np.diff(np.log(prices),n=1, axis=0)

	#load text data
	newsrootdir = "./Data/ReutersNews106521"
	newsdates = os.listdir(newsrootdir)
	news_data = []
	faulty_news = []
	sentences = []

	for i in newsdates:
		if not i.startswith('.'):
			newstitles = os.listdir(newsrootdir + "/" + i)
			for j in newstitles:
				if not j.startswith('.'):
					try:
						file = open(newsrootdir + "/" + i + "/" + j , "r") 
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
	
	mu = np.array([])
	sigma = np.array([])
	#weekly mean variance scaled..
	for i in range(len(lreturns)-n):
		np.append(mu, np.mean(lreturns[i:(i+n),:],axis=0))
		np.append(sigma, np.var(lreturns[i:(i+n),:],axis=0))

	if alone:
		f = open('./Data/processed_data', 'wb')
		pickle.dump([prices, dates, names, lreturns, news_data, faulty_news, 
			mu, sigma], f)
		f.close()

	return [prices, dates, names, lreturns, news_data, faulty_news,mu, sigma]
