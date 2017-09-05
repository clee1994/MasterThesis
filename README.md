# About
In 1952 Markowitz published his seminal paper on portfolio selection. In which he explained that the process of portfolio selection separates into two stages. In the first stage, an investor has to come up with beliefs about the future performance of the securities, in which it is possible to invest. The second stage is concerned about the formation of a portfolio based on those beliefs. However, for the first stage, Markowitz solely assumed that stock returns are random variables and investors somehow have probability beliefs about the securities. This assumption corresponds with Fama (1965). He argued that rational investors give assets an intrinsic value, which depends on the earnings prospect of a firm (which in turn depends on general or company-specific political and economic circumstances) and the prices fluctuate around this intrinsic value. Modern behavioural finance contradicts those finding by reasoning that irrational noise traders with incorrect stochastic beliefs due to emotional or cognitive biases influence prices substantially (De Long et al. 1990 and Nofsinger 2005). Nevertheless, both traditional finance and modern behavioural finance connect stock prices to information available to the investors. Despite the fact that many scholars already used financial news for stock price prediction, the information of financial news so far has not been widely used for portfolio formation purposes. This thesis aims to merge state-of-the-art machine learning techniques in natural language processing with portfolio theory. Moreover, the thesis intends to introduce a procedure to attain the required estimates of the mean vector and the covariance matrix for portfolio formation based on financial news and past prices. The procedure consists of three key steps. Firstly the mapping of financial news to numeric vectors amongst others via neural language model (paragraph to vector). Secondly, the learning task of predicting the return, variance and covariance based on the attained feature vectors and past prices (linear, ridge or support vector regression). Thirdly, building portfolios based on the estimates from step two. To the extent of the author’s knowledge, the application of neural language models (such as paragraph to vector) to obtain estimates for portfolio theory is unique. The results showed that the employment of certain neural language models was especially effective to achieve high alphas and Sharpe ratios and outperformed traditional methods which solely rely on past prices and most of the older feature creation methods. All portfolios based on the Reuters news headlines performed better than their counterpart of the traditional method which only built on past observations. Interestingly a subset of only ten stocks was sufficient to achieve these excellent results.

# Execution
At first you need to fetch the Reuters and SP500 Data and then you can execute the main script to run the portfolio optimization
```
bash fetchdata.sh
mkdir Output
mkdir Output/pics
mkdir Output/tables
python3 main.py
```
# References

Le, Q. & Mikolov, T. (2014), ‘Distributed Representations of Sentences and Documents’, International Conference on Machine Learning - ICML 2014 32, 1188–1196.
URL: http://arxiv.org/abs/1405.4053

Řehůřek, R. & Sojka, P. (2010), Software Framework for Topic Modelling with Large Corpora, in ‘Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks’, ELRA, Valletta, Malta, pp. 45–50.

Diamond, S. & Boyd, S. (2016), ‘CVXPY: A Python-Embedded Modeling Language for Convex Optimization’, Journal of Machine Learning Research 17(83), 1–5.

https://github.com/c0redumb/yahoo_quote_download

https://github.com/philipperemy/Reuters-full-data-set

