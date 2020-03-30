import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import minmax_scale


def dataset_stats(dataset):
    print(np.unique(dataset['sentiment']))
    print(np.bincount(dataset['sentiment']))


# Global Parameters:
dataset_fraction = 0.1


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
dataset['text'] = text
raw_target = pd.to_numeric(dataset['sentiment'])
target = minmax_scale(raw_target, feature_range=(-1.0, 1.0))
dataset['sentiment'] = target

print(dataset['timestamp'])
print(target)

vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, max_features=10000, strip_accents='unicode',lowercase=True,
                             analyzer='word',
                             # token_pattern=r"(?u)@?#?\b[\w]{2,}\b",
                             token_pattern=r"(?u)(?<![#@])\b[\w]{2,}\b",
                             ngram_range=(1, 1),
                             use_idf=True,
                             smooth_idf=True,
                             sublinear_tf=True,
                             binary=True,
                             norm=None,
                             stop_words='english'
                             )


print("Vectorizing.")
tfidf_data = vectorizer.fit_transform(text)
positive_text = dataset[dataset['sentiment'] == 1.0]['text']
negative_text = dataset[dataset['sentiment'] == -1.0]['text']
positive_text_xfrm = vectorizer.transform(positive_text)
negative_text_xfrm = vectorizer.transform(negative_text)

positive_text_inv = vectorizer.inverse_transform(positive_text_xfrm)
negative_text_inv = vectorizer.inverse_transform(negative_text_xfrm)

print("Apriori")

transaction_encoder = TransactionEncoder()
te_ary = transaction_encoder.fit(positive_text_inv).transform(positive_text_inv)
df = pd.DataFrame(te_ary, columns=transaction_encoder.columns_)
fi_pos = apriori(df, min_support=0.005, use_colnames=True, max_len=2)
fi_pos['length'] = fi_pos['itemsets'].apply(lambda x: len(x))
print(fi_pos[fi_pos['length'] >= 2])

transaction_encoder = TransactionEncoder()
te_ary = transaction_encoder.fit(negative_text_inv).transform(negative_text_inv)
df = pd.DataFrame(te_ary, columns=transaction_encoder.columns_)
fi_neg = apriori(df, min_support=0.005, use_colnames=True, max_len=2)
fi_neg['length'] = fi_neg['itemsets'].apply(lambda x: len(x))
print(fi_neg[fi_neg['length'] >= 2])

# Association Rules of postive frequent itemset
rules_pos = association_rules(fi_pos, metric="confidence", min_threshold=0.6)
plt.scatter(rules_pos['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

# Association Rules of postive frequent itemset
rules_neg = association_rules(fi_neg, metric="confidence", min_threshold=0.6)
plt.scatter(rules_neg['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
