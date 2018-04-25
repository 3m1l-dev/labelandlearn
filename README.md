# Label and Learn: Ethereum
Label and Learn is a simple Flask front-end intended for labelling data with the ability to retrain the algorithm using the data on the go. For this demo I have used tweets
about Ethereum, and have trained a simple neural net to identify spam and potentially useful tweets from large data-sets of tweets on Ethereum. The purpose of this is to collect
meaningful data for future sentiment analysis that can be used to predict Ethereum price fluctuations.

The project uses the following libraries:
- pandas
- numpy
- keras
- tensorflow
- base64
- urllib.parse

I have used hot-encoded vectors using 10000 words as the feature of each vector. This number can be changed around, however for larger values computation is longer. 
A simple neural net using dense layers and relu / sigmoid activations is used to predict a single output value of 0 or 1 for spam / potentially useful tweets. 
I've tokenised the words, done some preprocessing, and trained the basic classifier using the the most common words that appear in positive and negative tweets.
With this setup I'm getting around 96% accuracy for detecting useless tweets, and around 60% for detecting useful tweets.

I've made the Flask front-end very simple, with plans to expand it in the future. However for now it serves as an easy way to label, see the predictions of the
current classifier, and re-train when desired to update the predictions and see as the classifier improves over time. Currently I am only training one classifier
for detecting useful / useless tweets, however as you can see from the labels I am categorising tweets with their sentiment, as the next step after training a 
useful tweet classifier would be to scrape as many tweets as possible and train the sentiment analysis classifier.

![Screen-shot]('screen-shot.png')

There is great potential for this project however it has been taking long to develop as I have to label the data myself and I'm only working on this project
in my own time outside of work. However this inspired me to make a useful and efficient labelling and re-training front-end, and I am focusing on that first to 
make the rest of the project easier in the future. While doing this I realised the value there is in creating a front-end for labelling and training, as this can
be used in larger projects to spread data labelling work to anyone, as the front-end is very easy to use and requires no actual knowledge of coding.

I will potentially build this into a larger more generalised labelling and learning tool, and might want to incorporate online learning with stochastic gradient descent,
treating each tweet as a mini-batch and updating the weights live.

To run the web app, run the app.py as a python script, and open http://127.0.0.1:5000/ in your preferred web browser.

