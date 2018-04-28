import os
from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import base64
import urllib.parse
import io

nltk.download('stopwords')
application = app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Pre-processing function, with STOPWORDS imports.pated from nltk
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

# Correct predictions on useful and spam tweets
global correct_useful
global correct_spam

correct_useful = 0
correct_spam = 0

# Prediction accuracies
global acc_useful
global acc_spam

acc_useful = []
acc_spam = []

# Maximum number of words to use for feature vectors
max_words = 10000

# Images
images = os.path.join(APP_ROOT, 'images/')

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
def train():
	# Loading latest dataset
	source = os.path.join(APP_ROOT, 'data_sets/')
	data_set = "/".join([source, 'unlabelled.csv'])

	# Initializing global variables: dataframes, feature vectors, predictions and row iterator
	global df
	global training_vectors
	# Original copy of dataframe loaded from .csv file, used to display tweets
	global original
	# Array of predicted values for each tweet in new loaded dataframe
	global predicted
	global prediction
	# Re-initialise row to 0 if a new file is loaded
	row = 0

	# Pre-processing for predictions, loading the new file
	load = pd.read_csv(data_set)
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

	# Creating plot for accuracies, using BytesIO object to encode plot and pass to html
	img = io.BytesIO()
	plt.style.use('seaborn-whitegrid')
	plt.xlabel('Tweets Labelled')
	plt.ylabel('%')
	plt.legend([Line2D([0], [0], color='green', marker="o", ls='-', fillstyle='none', linewidth=0.5),
				Line2D([0], [0], color='red', marker="o", ls='-', fillstyle='none', linewidth=0.5)],
			   ['Accuracy on Useful', 'Accuracy on Spam'])
	plt.savefig(img, format='png')
	img.seek(0)
	plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())

	# Load the training page that displays tweet, date and prediction
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'], prediction=predicted[row],
						   plot_url = plot_url)


# Categories: Positive, Negative, Useless

# Both positive and negative tweets are useful, the reason they are differentiated at this point is to be used for
# sentiment analysis later on when I have finished collecting and labelling enough data.

@app.route('/label', methods=['POST'])
def label():
	global row
	global correct_useful
	global correct_spam
	global acc_useful
	global acc_spam
	global prediction

	# Updating the dataframe with ratings and values depending on value of submit button
	if request.method == 'POST':
		if request.form['submit'] == 'Positive':
			original.at[row, 'rating'] = 1
			original.at[row, 'useful'] = 1
			print(original.iloc[row])
			if round(prediction[row][0]) == 1:
				correct_useful += 1
			row += 1
		if request.form['submit'] == 'Negative':
			original.at[row, 'rating'] = -1
			original.at[row, 'useful'] = 1
			print(original.iloc[row])
			if round(prediction[row][0]) == 1:
				correct_useful += 1
			row += 1
		if request.form['submit'] == 'Useless':
			original.at[row, 'rating'] = 0
			original.at[row, 'useful'] = 0
			print(original.iloc[row])
			if round(prediction[row][0]) == 0:
				correct_spam += 1
			row += 1

		# Plotting accuracies for identifying useful and spam as function of total data points labelled.
		total_useful = original['useful'].iloc[:row].sum()
		total_spam = row - total_useful
		print(total_useful)
		print(total_spam)
		print(correct_useful)
		print(correct_spam)
		if total_useful == 0:
			acc_useful.append(0)
		else:
			acc_useful.append(100*(correct_useful/total_useful))
		if total_spam == 0:
			acc_spam.append(0)
		else:
			acc_spam.append((100*(correct_spam/total_spam)))
		print(acc_useful)
		print(acc_spam)

		# Use row + 1 as indexing starts from 0.
		img = io.BytesIO()
		x = np.arange(row)
		print(np.array(acc_useful))
		print(np.array(acc_spam))
		plt.plot(x, np.array(acc_useful), color='green', marker="o", ls='-', label='Accuracy on Useful', fillstyle='none')
		plt.plot(x, np.array(acc_spam), color='red', marker="o", ls='-', label='Accuracy on Spam', fillstyle='none')
		plt.legend([Line2D([0], [0], color='green', marker="o", ls='-', fillstyle='none'),
					Line2D([0], [0], color='red', marker="o", ls='-', fillstyle='none')], ['Accuracy on Useful', 'Accuracy on Spam'])
		plt.savefig(img, format='png')
		img.seek(0)
		plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())


	# Returns next tweet with prediction
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'],
						   prediction=predicted[row], plot_url=plot_url, acc_useful=round((100*(correct_useful/total_useful)), 1),
						   acc_spam=round((100*(correct_spam/total_spam)), 1))


@app.route('/re-train', methods=['POST'])
def retrain():
	labelled_so_far = original[:row]
	# Save remaining unlabelled as new file to keep track
	unlabelled = original[row:]
	target = os.path.join(APP_ROOT, 'data_sets/')
	unlabelled.to_csv(target + "demounlabelled.csv", encoding='utf-8', index=False)

	# Re-train model using new labels and existing data
	frames = [data, labelled_so_far]
	new_training_data = pd.concat(frames)
	# Save as new latest labelled data
	new_training_data.to_csv(target + "demo.csv", encoding='utf-8', index=False)
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
	with open('demo_model.json', 'w') as json_file:
		json_file.write(model_json)

	model.save_weights('demo_model.h5')

	# Use completely unseen data (untrained)

	print("Running Model on Test Set")

	prediction = model.predict(training_vectors[train_on:])
	actual_y = train_y[train_on:]
	total = len(actual_y)
	correct = 0
	useful = 0
	spam = 0
	global acc_useful
	global acc_spam
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

@app.route('/demo', methods=['POST', 'GET'])
def demo():
	# Loading latest dataset
	source = os.path.join(APP_ROOT, 'data_sets/')
	data_set = "/".join([source, 'demounlabelled.csv'])

	# Model structure created in neuralnetmodel.ipynb is laoded
	json_file = open('demo_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	# Create the model
	model = model_from_json(loaded_model_json)
	# Load the saved weights
	model.load_weights('demo_model.h5')
	model._make_predict_function()

	# Initializing global variables: dataframes, feature vectors, predictions and row iterator
	global df
	global training_vectors
	# Original copy of dataframe loaded from .csv file, used to display tweets
	global original
	# Array of predicted values for each tweet in new loaded dataframe
	global predicted
	global prediction
	# Re-initialise row to 0 if a new file is loaded
	global row
	row = 0
	# Correct predictions on useful and spam tweets
	global correct_useful
	global correct_spam

	correct_useful = 0
	correct_spam = 0

	# Prediction accuracies
	global acc_useful
	global acc_spam

	acc_useful = []
	acc_spam = []

	# Pre-processing for predictions, loading the new file
	load = pd.read_csv(data_set)
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

	img = io.BytesIO()
	plt.style.use('seaborn-whitegrid')
	plt.xlabel('Tweets Labelled')
	plt.ylabel('%')
	plt.legend([Line2D([0], [0], color='green', marker="o", ls='-', fillstyle='none', linewidth=0.5),
				Line2D([0], [0], color='red', marker="o", ls='-', fillstyle='none', linewidth=0.5)],
			   ['Accuracy on Useful', 'Accuracy on Spam'])
	plt.savefig(img, format='png')
	if row == 0:
		plt.clf()
		plt.cla()
		plt.close()
		plt.savefig(img, format='png')
	img.seek(0)
	plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())

	# Load the training page that displays tweet, date and prediction
	return render_template('train.html', text=original.iloc[row]['text'], date=original.iloc[row]['date'], prediction=predicted[row],
						   plot_url = plot_url)

# Run the app

if __name__ == '__main__':
	app.run()
