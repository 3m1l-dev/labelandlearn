# cryptotweets
Pulling old tweets and classifying them using a simple GUI, then feeding them to a classifier.

The TweetScrapeAndLabel part uses the below repository:
[Link](https://github.com/Jefferson-Henrique/GetOldTweets-python)

The file in TweetScrapeAndLabel that I made for the labeller is front.py, and also made some changes to
the user agent in TweetManager as user agents may get blocked depending on how many tweets you pull.
For that part I used Python 2.7, with Tkinter, pandas and numpy mostly.

Although I don't have enough data yet I made a quick preprocessing and classifier testing file just to showcase
the direction in which this project is going. This is the preprocessing.py file, where I used Python 3.6, nltk, pandas,
numpy, sklearn and some others. 
