# cryptotweets
This is a basic Flask app that I made as a part of a project to make labelling and re-training a small neural network to classify useful and useless tweets on Ethereum.
This part I am aiming to train to recognise the most valuable tweets to use for future price predictions. Valuable tweets in this case refer to tweets where sentimental
analysis can be performed on real (non-spam) tweets to calculate a daily overall rating for Ethereum as a cryptocurrency. I aim to use this rating in the future to plug
in as an indicator to a recurrent neural network along with the price data that can already be fetched from another API. I will then attempt to use this to predict price
fluctuations in the future.

I've tokenised the words, done some preprocessing, and trained a basic classifier using the the top 10,000 most common words that appear in positive and negative tweets.
With this setup I'm getting around 96% accuracy for detecting useless tweets, and around 36% for detecting useful tweets.

I've made the Flask front-end very simple, with plans to expand it in the future. However for now it serves as an easy way to label, see the predictions of the
current classifier, and re-train when desired to udpate the predictions and see as the classifier improves over time. Currently I am only training one classifier
for detecting useful / useless tweets, however as you can see from the labels I am categorising tweets with their sentiment, as the next step after training a 
useful tweet classifier would be to scrape as many tweets as possible and train the sentiment analysis classifier.

There is great potential for this project however it has been taking long to develop as I have to label the data myself and I'm only working on this project
in my own time outside of work. However this inspired me to make a useful and efficient labelling and re-training front-end, and I am focusing on that first to 
make the rest of the project easier in the future. While doing this I realised the value there is in creating a front-end for labelling and training, as this can
be used in larger projects to spread data labelling work to anyone, as the front-end is very easy to use and requires no actual knowledge of coding.

I will potentially build this into a larger more generalised labelling and learning tool, and might want to incorporate online learning with stochastic gradient descent,
treating each tweet as a mini-batch and updating the weights live.

To run the web app, run the app.py as a python script, and open http://127.0.0.1:5000/ in your preferred web browser. 

