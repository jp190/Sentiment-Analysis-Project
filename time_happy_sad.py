import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

import matplotlib.pyplot as plt


def dataset_stats(dataset):
    print(np.unique(dataset['sentiment']))
    print(np.bincount(dataset['sentiment']))


# Global Parameters:
dataset_fraction = 1.0


# Program:
print("Loading dataset.")
try:
    print("Loading from pickle.")
    data = pd.read_pickle('./data/frame.pkl')

    if dataset_fraction < 1.0:
        dataset = data.sample(frac=dataset_fraction, random_state=0)
    else:
        dataset = data

except:
    data = pd.read_csv("./data/training.1600000.processed.noemoticon.csv",
                       names=["sentiment", "id", "timestamp", "query", "user", "text"],
                       encoding="ISO-8859-1")

    if dataset_fraction < 1.0:
        dataset = data.sample(frac=dataset_fraction, random_state=0)
    else:
        dataset = data

    print("Loaded:", dataset.shape)
    print(dataset.keys())

    dataset_stats(dataset)

    print("Preprocessing data.")
    exclude_link_regex = r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    exclude_html_esc = r"&[\w#]+;"


    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])


dataset['hour'] = dataset['timestamp'].dt.hour
dataset['dayofweek'] = dataset['timestamp'].dt.dayofweek
raw_text = dataset['text']

raw_target = pd.to_numeric(dataset['sentiment'])
target = minmax_scale(raw_target, feature_range=(-1.0, 1.0))
dataset['sentiment'] = target
dataset.to_pickle('./data/frame.pkl')

sentiment_per_hour = dataset.groupby('hour').sentiment.agg(['mean'])
print(sentiment_per_hour)

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(sentiment_per_hour)
dataset.groupby(['hour','dayofweek']).agg('mean')['sentiment'].unstack().plot(ax=ax[0, 1])
dataset.groupby(['hour','dayofweek']).agg('sum')['sentiment'].unstack().plot(ax=ax[1, 0])
dataset.groupby(['hour','dayofweek']).agg('count')['sentiment'].unstack().plot(ax=ax[1, 1])


plt.show()
