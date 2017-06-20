from data_loading import load_news_data, load_SP_data 
from learning import build_word2vec_model, get_news_vector, gen_xy, nearest, rnn_model


#load and preprocess data
[news_data, faulty_news] = load_news_data(True)
[prices, dates, names, lreturns, mu, sigma,dates_SP_weekly] = load_SP_data(True)

model = build_word2vec_model(True) 

[aggregated_news, dates_news] = get_news_vector(True,model)

[x_train,y_train] = gen_xy(aggregated_news,mu,dates_news,dates_SP_weekly)

model1 = rnn_model(x_train,y_train)

#learn news effect

#train

#test


#minimum variance portfolio projected CG


#sparsity minimum variance


#performance testing
