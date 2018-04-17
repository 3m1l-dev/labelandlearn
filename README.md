# cryptotweets
Pulling old tweets and classifying them using a simple GUI, then feeding them to a classifier.

The TweetScrapeAndLabel part uses the following repository:
[Jefferson-Henrique: GetOldTweets-python](https://github.com/Jefferson-Henrique/GetOldTweets-python)

The file in TweetScrapeAndLabel that I made for the labeller is front.py, and also made some changes to
the user agent in TweetManager as user agents may get blocked depending on how many tweets you pull.
For the labeller part I used Python 2.7, with Tkinter, pandas and numpy mostly. This basic Tkinter app I made
to speed up the labelling process for myself initially so that I could build the first classifier. However, 
I dislike Tkinter and decided to move on to Flask. 

I've tokenised the words, done some preprocessing, and trained a basic classifier using the the top 10,000 most common words that appear in positive and negative tweets.
With this setup I'm getting around 96% accuracy for detecting useless tweets, and around 36% for detecting useful tweets.

I've made the Flask front-end very simple, with plans to expand it in the future. However for now it serves as an easy way to label, see the predictions of the
current classifier, and re-train when desired to udpate the predictions and see as the classifier improves over time. Currently I am only training one classifier
for detecting useful / useless tweets, however as you can see from the labels I am categorising tweets with their sentiment, as the next step after training a 
useful tweet classifier would be to scrape as many tweets as possible and train an Ethereum sentiment analysis classifier. This would be highly useful to study
the price of Ethereum and investigate the connection with the overall twitter sentiments of Ethereum that day and the price. A sentiment analysis could be fed in
as an indicator to a recurrent neural net to predict the price eventually.

There is great potential for this project however it has been taking long to develop as I have to label the data myself and I'm only working on this project
in my own time outside of work. However this inspired me to make a useful and efficient labelling and re-training front-end, and I am focusing on that first to 
make the rest of the project easier in the future. 

