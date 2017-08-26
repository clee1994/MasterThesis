
import evaluation
import learning

import datetime, pickle, gc, random
import numpy as np

path = '/Users/clemens/Desktop/complet'
path_data = 'Data/'

path_output = 'Output/'
learning.path_output = path_output
evaluation.path_output = path_output

number_jobs = 1
learning.number_jobs = number_jobs
evaluation.number_jobs = number_jobs

past_obs_int = False
learning.past_obs_int = past_obs_int

firms_used = 10
n_past = 120
n_past_add = 20
learning.n_past = n_past
learning.n_past_add = n_past_add
n_cov = 5
learning.n_cov = n_cov
test_split = 0.35

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)) and f[0]!='.')]

[final_complet, r4, r1, sp500] = pickle.load(open(path + "/" + onlyfiles[0] , "rb" ) )

for i in range(np.shape(onlyfiles)[0]-1):
	[complet,r4,r1,sp500] = pickle.load(open(path + "/" + onlyfiles[i+1] , "rb" ) )
	for j in range(np.shape(complet)[0]-1):
		print(complet[j+1][7])
		final_complet.append(complet[j+1])


[x_gram, dates_news] = pickle.load(open(path_output +"bowx_models4.p", "rb" ) )
split_point = int(np.floor(np.shape(x_gram[0])[0]*(1-test_split)))

[r4,sp500] = evaluation.pure_SP(dates_news[(split_point):],path_data)

evaluation.final_table(final_complet,np.array(r4),r1,sp500)

