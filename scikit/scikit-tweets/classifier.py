import pandas as pd
import numpy as np
import wandb

wandb.init()

# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('tweets.csv')

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']

# Get pandas Series object of the "emotion" column:
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

# Remove the blank rows from the series:
target = target[pd.notnull(text)]
text = text[pd.notnull(text)]

# Perform feature extraction:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(text)
counts = count_vect.transform(text)


counts_train = counts[:6000]
target_train = target[:6000]
counts_test = counts[6000:]
target_test = target[6000:]


# Train with this data with a Naive Bayes classifier:
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(counts, target)


X_test = counts_test
y_test = target_test
y_probas = nb.predict_proba(X_test)
y_pred = nb.predict(X_test)

print("y", y_probas.shape)

wandb.sklearn.plot_learning_curve(nb, X_test, y_test)
wandb.sklearn.plot_roc(y_test, y_probas)
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=nb.classes_)
wandb.sklearn.plot_precision_recall(y_test, y_probas)