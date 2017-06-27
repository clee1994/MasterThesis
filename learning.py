
#def learning_news_effect(alone, prices=0, dates=0, names=0, lreturns=0, 
#news_data=0)



def gen_dict(news):
	all_words = []
	for i in range(len(news)):
		for j in range(len(news[i][8])):
			for k in news[i][8][j]:
				all_words.append(k)

	prog_st = 0
	l = len(all_words) 
	printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)



	#gen dictionary
	mydict={}
	counter = 0
	for k in all_words:
		prog_st += 1
		printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

	    if(counter>0 and k in mydict):
	    	continue
	    else:
	    	counter +=1
	    	mydict[k] = [counter, all_words.count(k)]

	del all_words


	#sort by frequency
	sort_temp = sorted(mydict.items(), key=lambda x: x[1][1],reverse=True)
	del mydict

	prog_st = 0
	l = len(sort_temp) 
	printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


	sort_dict = []
	for i in range(len(sort_temp)):
		prog_st += 1
		printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

		sort_dict.append(sort_temp[i][0])
		#sort_dict[sort_temp[i][0]] = i


	return sort_dict



#def gen_xy_news(news,lreturns,dates_stocks,dict_words)




def gen_xy_daily(news,lreturns,dates_stocks,dict_words, words_used):
	import datetime
	import numpy as np
	from keras.preprocessing import sequence

	data_days = []
	y = []
	day_count = -1

	prog_st = 0
	l = len(news) 
	printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


	prev_d = np.datetime64(datetime.date(1, 1, 1))
	for i in news:
		prog_st += 1
		printProgressBar(prog_st, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


		temp_d = []
		temp = i[4].tolist()
		cur_d = np.datetime64(datetime.date(temp.year, temp.month, temp.day))

		if cur_d == prev_d:
			prev_d = cur_d
		else:
			#text/x
			prev_d = cur_d
			data_days.append([])
			day_count += 1
			#mu/y -> what do I want, the mean next day, average next three days
			try:
				y.append(lreturns[list(dates_stocks).index(cur_d),:])
			except:
				ind_temp = min(dates_stocks, key=lambda x: abs(x - cur_d))
				y.append(lreturns[list(dates_stocks).index(ind_temp),:])

		#skip last sentence
		for j in range(len(i[8])-1):
			for k in i[8][j]:
				data_days[day_count].append(dict_words.index(k))

	#length info
	#import numpy as np
	#lens = np.array(list(map(len, data_days)))
	#truncate at 15000

	x = sequence.pad_sequences(data_days, maxlen=15000, value=0)
	x[x > words_used] = words_used

	del data_days

	return x,np.array(y)



def train_test_split(x,y,test_split):
	import numpy as np
	split_point = int(np.floor(np.shape(x)[0]*(1-test_split)))
	x_train = x[0:split_point,:]
	y_train = y[0:split_point,:]
	x_test = x[(split_point+1):,:]
	y_test = y[(split_point+1):,:]
	return x_train,y_train,x_test,y_test



