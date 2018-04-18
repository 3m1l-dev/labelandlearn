import os
from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Pre-processing function, with STOPWORDS imported from nltk
STOPWORDS = set(stopwords.words('english'))

def format_text(s):
	s = re.sub(r"http\S+", "", s)
	s = re.sub('[^0-9a-z #+_]', ' ', s.lower());
	s = " ".join(word for word in s.split() if word not in STOPWORDS)
	return s

def get_feature_vectors(df):
	df.loc[:, 'text'] = df.text.apply(lambda x: format_text(x))
	df.loc[:, 'text'] = df.text.apply(lambda x: " ".join(re.findall('[\w]+', x)))
	text_only = df['text']
	texts = np.asarray(text_only.values)
	# Tokenizing
	tknzr.fit_on_texts(texts)
	tokenized_train_x = tknzr.texts_to_sequences(texts)

	# Remove duplicate tokens
	for i in range(0, len(tokenized_train_x)):
		tokenized_train_x[i] = list(set(tokenized_train_x[i]))
	training_vectors = np.zeros((len(tokenized_train_x), max_words))

	# Create one-hot matrices out of the indexed tweets
	for i in range(0, len(tokenized_train_x)):
		training_vectors[i][tokenized_train_x[i]] = 1
	return training_vectors


# Load labelled data, model, variables

# Row variable used to iterate over data points in dataframe
global row
row = 0
# Maximum number of words to use for feature vectors
max_words = 10000

# The latest set of labelled data that the network is trained on is loaded from the main project folder
data_sets = os.path.join(APP_ROOT, 'data_sets/')
data = pd.read_csv(data_sets + 'latest.csv')
latest = data[['useful', 'text']]

# Model structure created in neuralnetmodel.ipynb is laoded
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# Create the model
model = model_from_json(loaded_model_json)
# Load the saved weights
model.load_weights('model.h5')
model._make_predict_function()

# Tokenizer
tknzr = Tokenizer(lower=True, split=" ", num_words=max_words)


@app.route('/', methods=['POST', 'GET'])
def home():
	return render_template('home.html')


@app.route('/train', methods=['POST'])
def train():
	# Uploading and saving new data sets
	target = os.path.join(APP_ROOT, 'data_sets/')

	if not os.path.isdir(target):
		os.mkdir(target)

	for file in request.files.getlist('file'):
		# Saving file
		filename = file.filename
		destination = "/".join([target, filename])
		print(destination)
		file.save(destination)

	# Initializing global variables: dataframes, feature vectors, predictions and row iterator
	global df
	global training_vectors
	# Original copy of dataframe loaded from .csv file, used to display tweets
	global original
	# Array of predicted values for each tweet in new loaded dataframe
	global predicted
	# Re-initialise row to 0 if a new file is loaded
	row = 0

	# Pre-processing for predictions, loading the new file
	load = pd.read_csv(destination)
	# Null values removed to avoid classifier malfunction
	df = load[load['text'].notnull()]
	original = df.copy()

	training_vectors = get_feature_vectors(df)

	# Make predictions
	prediction = model.predict(training_vectors)
	predicted = []
	for p in range(0, len(prediction)):
		if round(prediction[p][0]) == 1:
			predicted.append("1: Potentially Useful")
		if round(prediction[p][0]) == 0:
			predicted.append("0: Useless / Spam")

	# Load the training page that displays tweet, date and prediction
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'], prediction=predicted[row])


# Categories: Positive, Negative, Useless

# Both positive and negative tweets are useful, the reason they are differentiated at this point is to be used for
# sentiment analysis later on when I have finished collecting and labelling enough data.

@app.route('/label', methods=['POST'])
def label():
	global row
	# Updating the dataframe with ratings and values depending on value of submit button
	if request.method == 'POST':
		if request.form['submit'] == 'Positive':
			original.at[row, 'rating'] = 1
			original.at[row, 'useful'] = 1
			print(original.iloc[row])
			row += 1
		if request.form['submit'] == 'Negative':
			original.at[row, 'rating'] = -1
			original.at[row, 'useful'] = 1
			print(original.iloc[row])
			row += 1
		if request.form['submit'] == 'Useless':
			original.at[row, 'rating'] = 0
			original.at[row, 'useful'] = 0
			print(original.iloc[row])
			row += 1
	# Returns next tweet with prediction
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'],
						   prediction=predicted[row])


@app.route('/re-train', methods=['POST'])
def retrain():
	labelled_so_far = original[:row]
	# Save remaining unlabelled as new file to keep track
	unlabelled = original[row:]
	target = os.path.join(APP_ROOT, 'data_sets/')
	unlabelled.to_csv(target + "unlabelled.csv", encoding='utf-8', index=False)

	# Re-train model using new labels and existing data
	frames = [data, labelled_so_far]
	new_training_data = pd.concat(frames)
	# Save as new latest labelled data
	new_training_data.to_csv(target + "latest.csv", encoding='utf-8', index=False)
	# Get feature vectors for training
	training_vectors = get_feature_vectors(new_training_data)
	train_y = new_training_data["useful"].values

	data_points = len(new_training_data.index)

	train_on = int(0.9*data_points)

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(training_vectors[:train_on], train_y[:train_on],
			  batch_size=32,
			  epochs=10,
			  verbose=1,
			  validation_split=0.1,
			  shuffle=True)

	model_json = model.to_json()
	with open('model.json', 'w') as json_file:
		json_file.write(model_json)

	model.save_weights('model.h5')

	# Use completely unseen data (untrained)

	print("Running Model on Test Set")

	prediction = model.predict(training_vectors[train_on:])
	actual_y = train_y[train_on:]
	total = len(actual_y)
	correct = 0
	useful = 0
	spam = 0

	actual_useful = np.count_nonzero(actual_y)
	actual_spam = len(actual_y) - actual_useful


	print("Number of spam tweets: " + str(actual_spam) + " Number of useful tweets: " + str(actual_useful))

	for p in range(0, len(prediction)):
		predicted = round(prediction[p][0])
		if predicted == actual_y[p]:
			correct += 1
			if predicted == 1:
				useful += 1
			if predicted == 0:
				spam += 1

	p_acc = round((100*(correct/total)), 1)
	p_identified_useful = round((100*(useful/actual_useful)), 1)
	p_identified_spam = round((100*(spam/actual_spam)), 1)

	print("Accuracy on test set:  " + str(correct / total))
	print("Identified " + str(useful / actual_useful) + " of useful tweets")
	print("Identified " + str(spam / actual_spam) + " of spam tweets")
	return render_template('retrain.html', ac_spam=str(actual_spam), ac_useful=str(actual_useful), acc=str(p_acc),
						   identified_useful=str(p_identified_useful), identified_spam=str(p_identified_spam))

# Run the app

if __name__ == '__main__':
	app.run()
