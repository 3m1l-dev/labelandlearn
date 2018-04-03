# cryptotweets
Pulling old tweets and classifying them using a simple GUI, then feeding them to a classifier.

The TweetScrapeAndLabel part uses the following repository:
[Jefferson-Henrique: GetOldTweets-python](https://github.com/Jefferson-Henrique/GetOldTweets-python)

The file in TweetScrapeAndLabel that I made for the labeller is front.py, and also made some changes to
the user agent in TweetManager as user agents may get blocked depending on how many tweets you pull.
For the labeller part I used Python 2.7, with Tkinter, pandas and numpy mostly.

Although I don't have enough data yet I made a quick preprocessing and classifier testing file just to showcase
the direction in which this project is going. This is the preprocessing.py file, where I used Python 3.6, nltk, pandas,
numpy, sklearn and some others. Here I've tokenised the words, done some preprocessing, and trained a few classifiers using
the top 10,000 most common words that appear in positive and negative tweets. Even with this basic setup I'm getting around 85% 
accuracy.

The next stages (after I collect and label at least 20k tweets manually) involve building a neural net for the classification,
doing some tuning and aiming for a very high accuracy so that I can start pulling tweets and labelling them at the same time. I aim to combine
the classifier and labeller, but I will probably have to write my own version of the tweet puller part in Python 3.6.

The reason the project has been taking long to develop is mostly due to the fact that I have to label the data myself and I'm only working on this project
in my own time outside of work. Still the main aim would be to train a very accurate meaningful tweet fetcher for cryptos, as this data could be used for 
predicting fluctuations in stock price. This could be done through linking the labeller to the Twitter API and using positive / negative analysis on live tweets about a crypto currency
as an indicator to a RNN to predict crypto prices.....

