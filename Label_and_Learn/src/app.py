import os
from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from keras.models import model_from_json

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Pre-processing
STOPWORDS = set(stopwords.words('english'))


def format_text(s):
	s = re.sub(r"http\S+", "", s)
	s = re.sub('[^0-9a-z #+_]', ' ', s.lower());
	s = " ".join(word for word in s.split() if word not in STOPWORDS)
	return s

#def predict(x):
#	print(training_vectors[row].shape)
#	prediction = round(model.predict(training_vectors[x]))
#	print(prediction)
#	print("hello")
#	if prediction == 0:
#		text = "Useless"
#	else:
#		text = "Potentially Useful"
#	return prediction, text



# Load labelled data, model, variables

global row
row = 0
max_words = 10000

data = pd.read_csv('latest.csv')
latest = data[['useful', 'text']]

# Load model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')
model._make_predict_function()

# Tokenizer
tknzr = Tokenizer(lower=True, split=" ", num_words=max_words)




@app.route('/')
def home():
	return render_template('home.html')


@app.route('/train', methods=['POST'])
def train():
	target = os.path.join(APP_ROOT, 'data_sets/')
	print(target)

	if not os.path.isdir(target):
		os.mkdir(target)

	for file in request.files.getlist('file'):
		global training_vectors
		global df
		global original
		global predicted
		row = 0

		# Saving file
		filename = file.filename
		destination = "/".join([target, filename])
		print(destination)
		file.save(destination)

	# Pre-processing for predictions
	load = pd.read_csv(destination)
	df = load[load['text'].notnull()]
	original = df.copy()
	df.loc[:, 'text'] = df.text.apply(lambda x: format_text(x))
	df.loc[:, 'text'] = df.text.apply(lambda x: " ".join(re.findall('[\w]+', x)))
	text_only = df['text']
	texts = np.asarray(text_only.values)
	print(texts)
	tknzr.fit_on_texts(texts)
	tokenized_train_x = tknzr.texts_to_sequences(texts)

	# remove duplicate tokens
	for i in range(0, len(tokenized_train_x)):
		tokenized_train_x[i] = list(set(tokenized_train_x[i]))
	training_vectors = np.zeros((len(tokenized_train_x), max_words))

	# create one-hot matrices out of the indexed tweets
	for i in range(0, len(tokenized_train_x)):
		training_vectors[i][tokenized_train_x[i]] = 1
	print("OK")
	prediction = model.predict(training_vectors)
	print("swag")
	predicted = []
	for p in range(0, len(prediction)):
		if round(prediction[p][0]) == 1:
			predicted.append("1: Potentially Useful")
		if round(prediction[p][0]) == 0:
			predicted.append("0: Useless / Spam")

	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'], prediction=predicted[row])


@app.route('/positive', methods=['POST'])
def positive():
	global row
	original.at[row, 'rating'] = 1
	original.at[row, 'useful'] = 1
	print(original.iloc[row])
	row += 1
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'], prediction=predicted[row])

@app.route('/negative', methods=['POST'])
def negative():
	global row
	original.at[row, 'rating'] = 1
	original.at[row, 'useful'] = 1
	print(original.iloc[row])
	row += 1
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'], prediction=predicted[row])

@app.route('/useless', methods=['POST'])
def useless():
	global row
	original.at[row, 'rating'] = 1
	original.at[row, 'useful'] = 1
	print(original.iloc[row])
	row += 1
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'], prediction=predicted[row])


if __name__ == '__main__':
	app.run()
