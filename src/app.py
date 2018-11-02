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

# Load labelled data, model, variables

# Row variable used to iterate over data points in dataframe
ROW = 0

# Correct predictions on useful and spam tweets
CORRECT_USEFUL = 0
CORRECT_SPAM = 0

# Prediction accuracies
ACC_USEFUL = []
ACC_SPAM = []
# Maximum number of words to use for feature vectors
MAX_WORDS = 10000

# Images
IMAGES = os.path.join(APP_ROOT, 'images/')

# Latest set of data loaded from the main project folder
DATA_SETS = os.path.join(APP_ROOT, 'data_sets/')
DATA = pd.read_csv(DATA_SETS + 'latest.csv')
LATEST = DATA[['useful', 'text']]

# Model structure created in neuralnetmodel.ipynb is laoded
JSON_FILE = open('model.json', 'r')
LOADED_MODEL_JSON = JSON_FILE.read()
JSON_FILE.close()

# Create the model
MODEL = model_from_json(LOADED_MODEL_JSON)

# Load the saved weights
MODEL.load_weights('model.h5')
MODEL._make_predict_function()

# Tokenizer
TKNZR = Tokenizer(lower=True, split=" ", num_words=MAX_WORDS)

def format_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub('[^0-9a-z #+_]', ' ', text.lower())
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


def get_feature_vectors(data_frame):
    data_frame['text'] = [format_text(x) for x in data_frame['text'].values]
    data_frame['text'] = data_frame.text.apply(
        lambda x: " ".join(re.findall(r'[\w]+', x)))
    text_only = data_frame['text']
    texts = np.asarray(text_only.values)
    # Tokenizing
    TKNZR.fit_on_texts(texts)
    tokenized_train_x = TKNZR.texts_to_sequences(texts)


    training_vectors = np.zeros((len(tokenized_train_x), MAX_WORDS))
    # Remove duplicate tokens
    for val in enumerate(tokenized_train_x):
        tokenized_train_x[val[0]] = list(set(tokenized_train_x[val[0]]))
        # Create one-hot matrices out of the indexed tweets
        training_vectors[val[0]][tokenized_train_x[val[0]]] = 1

    return training_vectors


@app.route('/', methods=['POST', 'GET'])
def train():
    # Set global variables

    global original
    global ROW
    global prediction
    global predicted

    # Loading latest dataset
    source = os.path.join(APP_ROOT, 'data_sets/')
    data_set = "/".join([source, 'unlabelled.csv'])

    # Re-initialise row to 0 if a new file is loaded
    ROW = 0

    # Pre-processing for predictions, loading the new file
    load = pd.read_csv(data_set)
    # Null values removed to avoid classifier malfunction
    df = load[load['text'].notnull()]
    
    original = df.copy()

    training_vectors = get_feature_vectors(df)

    # Make predictions
    prediction = MODEL.predict(training_vectors)

    predicted = []
    for p in range(0, len(prediction)):
        if round(prediction[p][0]) == 1:
            predicted.append("1: Potentially Useful")
        if round(prediction[p][0]) == 0:
            predicted.append("0: Useless / Spam")

    # Creating plot for accuracies, using BytesIO object to encode plot and
    # pass to html
    img = io.BytesIO()
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Tweets Labelled')
    plt.ylabel('%')
    plt.legend([Line2D([0],
                       [0],
                       color='green',
                       marker="o",
                       ls='-',
                       fillstyle='none',
                       linewidth=0.5),
                Line2D([0],
                       [0],
                       color='red',
                       marker="o",
                       ls='-',
                       fillstyle='none',
                       linewidth=0.5)],
               ['Accuracy on Useful',
                'Accuracy on Spam'])
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())

    # Load the training page that displays tweet, date and prediction
    return render_template(
        'train.html',
        text=original.iloc[ROW]['text'],
        date=original.iloc[ROW]['date'],
        prediction=predicted[ROW],
        plot_url=plot_url)


# Categories: Positive, Negative, Useless

# Both positive and negative tweets are useful, the reason they are differentiated at this point is to be used for
# sentiment analysis later on when I have finished collecting and
# labelling enough data.

@app.route('/label', methods=['POST'])
def label():
    # Updating the dataframe with ratings and values depending on value of
    # submit button
    global ROW
    global CORRECT_USEFUL
    global CORRECT_SPAM
    global ACC_USEFUL
    global ACC_SPAM
    if request.method == 'POST':
        if request.form['submit'] == 'Positive':
            original.at[ROW, 'rating'] = 1
            original.at[ROW, 'useful'] = 1
            print(original.iloc[ROW])
            if round(prediction[ROW][0]) == 1:
                CORRECT_USEFUL += 1
            ROW += 1
        if request.form['submit'] == 'Negative':
            original.at[ROW, 'rating'] = -1
            original.at[ROW, 'useful'] = 1
            print(original.iloc[ROW])
            if round(prediction[ROW][0]) == 1:
                CORRECT_USEFUL += 1
            ROW += 1
        if request.form['submit'] == 'Useless':
            original.at[ROW, 'rating'] = 0
            original.at[ROW, 'useful'] = 0
            print(original.iloc[ROW])
            if round(prediction[ROW][0]) == 0:
                CORRECT_SPAM += 1
            ROW += 1

        # Plotting accuracies for identifying useful and spam as function of
        # total data points labelled.
        total_useful = original['useful'].iloc[:ROW].sum()
        total_spam = ROW - total_useful
        print(total_useful)
        print(total_spam)
        print(CORRECT_USEFUL)
        print(CORRECT_SPAM)
        if total_useful == 0:
            ACC_USEFUL.append(0)
        else:
            ACC_USEFUL.append(100 * (CORRECT_USEFUL / total_useful))
        if total_spam == 0:
            ACC_SPAM.append(0)
        else:
            ACC_SPAM.append((100 * (CORRECT_SPAM / total_spam)))
        print(ACC_USEFUL)
        print(ACC_SPAM)

        # Use row + 1 as indexing starts from 0.
        img = io.BytesIO()
        x = np.arange(ROW)
        print(np.array(ACC_USEFUL))
        print(np.array(ACC_SPAM))
        plt.plot(
            x,
            np.array(ACC_USEFUL),
            color='green',
            marker="o",
            ls='-',
            label='Accuracy on Useful',
            fillstyle='none')
        plt.plot(
            x,
            np.array(ACC_SPAM),
            color='red',
            marker="o",
            ls='-',
            label='Accuracy on Spam',
            fillstyle='none')
        plt.legend([Line2D([0],
                           [0],
                           color='green',
                           marker="o",
                           ls='-',
                           fillstyle='none'),
                    Line2D([0],
                           [0],
                           color='red',
                           marker="o",
                           ls='-',
                           fillstyle='none')],
                   ['Accuracy on Useful',
                    'Accuracy on Spam'])
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())

    # Returns next tweet with prediction
    return render_template('train.html',
                           text=original.iloc[ROW]['text'],
                           date=original.iloc[ROW]['date'],
                           prediction=predicted[ROW],
                           plot_url=plot_url,
                           acc_useful=round((100 * (CORRECT_USEFUL / total_useful)),
                                            1),
                           acc_spam=round((100 * (CORRECT_SPAM / total_spam)),
                                          1))


@app.route('/re-train', methods=['POST'])
def retrain():

    global ROW
    global CORRECT_USEFUL
    global CORRECT_SPAM
    global ACC_USEFUL
    global ACC_SPAM

    labelled_so_far = original[:ROW]
    # Save remaining unlabelled as new file to keep track
    unlabelled = original[ROW:]
    target = os.path.join(APP_ROOT, 'data_sets/')
    unlabelled.to_csv(
        target +
        "demounlabelled.csv",
        encoding='utf-8',
        index=False)

    # Re-train model using new labels and existing data
    frames = [DATA, labelled_so_far]
    new_training_data = pd.concat(frames)
    # Save as new latest labelled data
    new_training_data.to_csv(
        target + "demo.csv",
        encoding='utf-8',
        index=False)
    # Get feature vectors for training
    training_vectors = get_feature_vectors(new_training_data)
    train_y = new_training_data["useful"].values

    data_points = len(new_training_data.index)

    train_on = int(0.9 * data_points)

    MODEL.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    MODEL.fit(training_vectors[:train_on], train_y[:train_on],
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_split=0.1,
              shuffle=True)

    model_json = MODEL.to_json()
    with open('demo_model.json', 'w') as json_file:
        json_file.write(model_json)

    MODEL.save_weights('demo_model.h5')

    # Use completely unseen data (untrained)

    print("Running Model on Test Set")

    prediction = MODEL.predict(training_vectors[train_on:])
    actual_y = train_y[train_on:]
    total = len(actual_y)
    correct = 0
    useful = 0
    spam = 0
    actual_useful = np.count_nonzero(actual_y)
    actual_spam = len(actual_y) - actual_useful

    print(
        "Number of spam tweets: " +
        str(actual_spam) +
        " Number of useful tweets: " +
        str(actual_useful))

    for p in range(0, len(prediction)):
        predicted = round(prediction[p][0])
        if predicted == actual_y[p]:
            correct += 1
            if predicted == 1:
                useful += 1
            if predicted == 0:
                spam += 1

    p_acc = round((100 * (correct / total)), 1)
    p_identified_useful = round((100 * (useful / actual_useful)), 1)
    p_identified_spam = round((100 * (spam / actual_spam)), 1)

    print("Accuracy on test set:  " + str(correct / total))
    print("Identified " + str(useful / actual_useful) + " of useful tweets")
    print("Identified " + str(spam / actual_spam) + " of spam tweets")
    return render_template(
        'retrain.html',
        ac_spam=str(actual_spam),
        ac_useful=str(actual_useful),
        acc=str(p_acc),
        identified_useful=str(p_identified_useful),
        identified_spam=str(p_identified_spam))


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

    row = 0
    # Correct predictions on useful and spam tweets
    correct_useful = 0
    correct_spam = 0


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
    plt.legend([Line2D([0],
                       [0],
                       color='green',
                       marker="o",
                       ls='-',
                       fillstyle='none',
                       linewidth=0.5),
                Line2D([0],
                       [0],
                       color='red',
                       marker="o",
                       ls='-',
                       fillstyle='none',
                       linewidth=0.5)],
               ['Accuracy on Useful',
                'Accuracy on Spam'])
    plt.savefig(img, format='png')
    if ROW == 0:
        plt.clf()
        plt.cla()
        plt.close()
        plt.savefig(img, format='png')
    img.seek(0)
    plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())

    # Load the training page that displays tweet, date and prediction
    return render_template(
        'train.html',
        text=original.iloc[ROW]['text'],
        date=original.iloc[ROW]['date'],
        prediction=predicted[ROW],
        plot_url=plot_url)

# Run the app


if __name__ == '__main__':
    app.run()
