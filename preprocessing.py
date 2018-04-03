import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
import numpy as np
import re
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify import ClassifierI
from statistics import mode


class SelectClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# Import dataset, tokenize all words, get frequencies
label = pd.read_csv("jan01and04.csv")

# Preprocessing
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(s):
    s = " ".join(word for word in s.split() if word not in STOPWORDS)
    return s

label = label[label["text"].notnull()]
label.loc[:,"text"] = label.text.apply(lambda x: remove_stopwords(x))
label.loc[:, "text"] = label.text.apply(lambda x : " ".join(re.findall('[\w]+'
         ,x)))
label["text"] = label["text"].str.lower()

wordstext = label["text"].str.cat(sep=" ")
tokens = word_tokenize(wordstext)

all_words = []
for w in tokens:
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)

# Get 5000 most common words from tweets
word_features = [w[0] for w in sorted(all_words.items(), key=lambda k_v:k_v[1],
                 reverse=True)[:10000]]

# Format for training
tokens_label = label[['text', 'useful']]
tokens_label["tokenized_texts"] = tokens_label["text"].fillna("").map(
        nltk.word_tokenize)
features = tokens_label.drop(columns=['text'])
feature_tuples = [tuple(x) for x in features.values]

def get_features(document):
    features = {}
    for w in word_features:
        features[w] = (w in document)
    return features

featuresets = [(x, get_features(y)) for (x, y) in feature_tuples]
training_features = [(y, x) for (x, y) in featuresets]

training_set = training_features[:3200]
testing_set = training_features[3200:]

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy:", (nltk.classify.accuracy(MNB_classifier,
                                                          testing_set))*100)

Bernoulli_classifier = SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print("Bernoulli_classifier accuracy:",
      (nltk.classify.accuracy(Bernoulli_classifier, testing_set))*100)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD_classifier accuracy:",
      (nltk.classify.accuracy(SGD_classifier, testing_set))*100)

selected_classifier = SelectClassifier(MNB_classifier,
                                       Bernoulli_classifier, SGD_classifier)
print("Best classifier accuracy:",
      (nltk.classify.accuracy(selected_classifier, testing_set))*100)
print("Classification:", selected_classifier.classify(testing_set[0][0]),
      "Confidence:", selected_classifier.confidence(testing_set[0][0]))
    
# Pickle

#save_selected_classifier = open("Bernoulli_classifier.pickle", "wb")
#pickle.dump(Bernoulli_classifier, save_selected_classifier)
#save_selected_classifier.close()
#
#classifier_f = open("Bernoulli_classifier.pickle", "rb")
#Bernoulli_classifier = pickle.load(classifier_f)
#classifier_f.close()



