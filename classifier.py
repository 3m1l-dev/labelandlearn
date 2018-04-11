import numpy as np
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

df = pd.read_csv("5001labelled.csv")
label = df[['useful', 'text']]

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
training = [tuple(x) for x in label.values]


#training = np.genfromtxt(r'C:/Users/embuu/cryptotweets/5001labelled.csv', delimiter=',', skip_header=1, usecols=(2, 3), dtype=None)
# create our training data from the tweets
train_x = [x[1] for x in training]
# index all the sentiment labels
train_y = np.asarray([x[0] for x in training])

# only work with the 3000 most popular words found in our dataset
max_words = 10000

# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)
    
def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 2)

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
  metrics=['accuracy'])

model.fit(train_x, train_y,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')