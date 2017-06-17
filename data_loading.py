import numpy as np
import pandas as pd
import os as os
import pickle

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
					news_data.append([np.datetime64(i),j,title, author, \
						date, url, text])
				except ValueError:
					faulty_news.append([i,j])

#weekly mean variance using 



f = open('./Data/processed_data', 'wb')
pickle.dump([prices, dates, names, lreturns, news_data, faulty_news], f)
f.close()
