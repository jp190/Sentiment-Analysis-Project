import pandas as pd

import numpy as np
from scipy.sparse import csr_matrix, find
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def dataset_stats(dataset):
    print(np.unique(dataset['sentiment']))
    print(np.bincount(dataset['sentiment']))


# Global Parameters:
dataset_fraction = 0.1
test_fraction = 0.8


# Program:
print("Loading dataset.")
data = pd.read_csv("./data/training.1600000.processed.noemoticon.csv",
                   names=["sentiment", "id", "timestamp", "query", "user", "text"],
                   encoding="ISO-8859-1",)

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

raw_text = dataset['text']
text = raw_text.str.replace(exclude_link_regex, "")
text = text.str.replace(exclude_html_esc, "")
raw_target = pd.to_numeric(dataset['sentiment'])
target = minmax_scale(raw_target, feature_range=(-1.0, 1.0))

print("A:", dataset[text.str.contains('.*ice cream.*')].values)

print(target)

vectorizer = TfidfVectorizer(min_df=0.0, max_df=0.5, max_features=100000, strip_accents='unicode',lowercase=True,
                             analyzer='word',
                             # token_pattern=r"(?u)@?#?\b[\w]{2,}\b",
                             token_pattern=r"(?u)(?<![#@])\b[\w]{2,}\b",
                             ngram_range=(1, 3),
                             use_idf=True,
                             smooth_idf=True,
                             sublinear_tf=True,
                             norm=None,
                             stop_words='english'
                             )


print("Vectorizing.")
tfidf_data = vectorizer.fit_transform(text)

# count_vectorizer = CountVectorizer(min_df=0.0, max_df=0.3, max_features=100000, strip_accents='unicode',lowercase=True,
#                                    analyzer='word',
#                                    # token_pattern=r"(?u)@?#?\b[\w]{2,}\b",
#                                    token_pattern=r"(?u)(?<![#@])\b[\w]{2,}\b",
#                                    ngram_range=(1, 2),
#                                    stop_words='english',
#                                    vocabulary=vectorizer.vocabulary_
#                                  )

# count_data = count_vectorizer.fit_transform(text)
# print(count_data)
# print(vectorizer.get_feature_names(), count_vectorizer.get_feature_names())
print("Stop words:", len(vectorizer.stop_words_))

# print("STOP WORDS:", vectorizer.stop_words_)
print("TFIDF:", type(tfidf_data), tfidf_data)
print("IDF:", vectorizer.idf_)

# Compute some metrics per word:
target=target.reshape((target.shape[0], 1))
print("SHP:", tfidf_data.shape, target.shape)
word_scores_by_doc = tfidf_data.multiply(target)
print("WSbD:", word_scores_by_doc)


def train_test_lr(data, target, test_frac):
    classifiers = {"LR": LogisticRegression(),
                   "SGD": SGDClassifier(),
                   # "GNB": GaussianNB(),
                   # "MNB": MultinomialNB()
                   }

    X_train, X_test, y_train, y_test =  train_test_split(data, target,
                                                        test_size=test_fraction,
                                                        random_state=0)
    print("Dataset:", )
    print("Train/test sets:", X_test.shape, X_test.shape, y_train.shape, y_test.shape)
    for name, classifier in classifiers.items():
        print("Model:", name)

        classifier.fit(X_train, y_train.ravel())

        print("  Training:", X_test.shape)
        # print(vectorizer.inverse_transform(X_test))
        y_pred = classifier.predict(X_test)
        print("  Pred:", accuracy_score(y_true=y_test, y_pred=y_pred))

def train_test_lr_for_new_sentences(data, target, test_frac):
    classifier = SGDClassifier()

    X_train, X_test, y_train, y_test =  train_test_split(data, target,
                                                        test_size=test_fraction,
                                                        random_state=0)
    classifier.fit(X_train, y_train.ravel())

    y_pred = classifier.predict(X_test)
    print("  Pred:", accuracy_score(y_true=y_test, y_pred=y_pred))
    return classifier


print("Training models:")

print("TFIDF model:")
train_test_lr(tfidf_data, target, test_fraction)

print("Word scores model:")
train_test_lr(word_scores_by_doc, target, test_fraction)

model = train_test_lr_for_new_sentences(tfidf_data, target, test_fraction)
while True:
    sentence = input("Enter a sentence:")
    transform_sentence = vectorizer.transform([sentence])
    print(sentence)
    print(transform_sentence)
    print(model.predict(transform_sentence))

