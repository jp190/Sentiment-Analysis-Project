from pandas import read_csv
import numpy as np
from bokeh.client import push_session
from bokeh.io import curdoc
from datashader.tests.test_bokeh_ext import create_image
from scipy.sparse import csr_matrix, find
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import minmax_scale
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import holoviews as hv
import holoviews.operation.datashader as hd
import bokeh.plotting as bp
from datashader.bokeh_ext import InteractiveImage
from datashader.utils import export_image
hd.shade.cmap=["lightblue", "darkblue"]
hv.extension("bokeh", "matplotlib")


def dataset_stats(dataset):
    print(np.unique(dataset['sentiment']))
    print(np.bincount(dataset['sentiment']))


# Global Parameters:
dataset_fraction = 0.1
test_fraction = 0.8


# Program:
print("Loading dataset.")
data = read_csv("./data/training.1600000.processed.noemoticon.csv",
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

vectorizer = TfidfVectorizer(min_df=0.0, max_df=0.5, max_features=10000, strip_accents='unicode',lowercase=True,
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


def non_zero_mean(mtx):
    (x, y, z) = find(mtx)
    countings = np.bincount(y)
    sums = np.bincount(y, weights=z)
    mean = sums / countings
    return mean


def non_zero_variance(mtx):
    mtx_sq = mtx.power(2.0)
    val_sq_exp_val = non_zero_mean(mtx_sq)
    exp_val_sq = np.power(non_zero_mean(mtx), 2.0)
    var = val_sq_exp_val - exp_val_sq
    var *= (var > 0)
    return var


def non_zero_stdev(mtx):
    var = non_zero_variance(mtx)
    stdev = np.sqrt(var)
    return stdev


def non_zero_norm(mtx, min_val=-1, max_val=1):
    d = mtx.data
    res = csr_matrix(mtx)
    res_data = res.data

    lens = mtx.getnnz(axis=1)
    idx = np.r_[0, lens[:-1].cumsum()]

    maxs = np.maximum.reduceat(d, idx)
    mins = np.minimum.reduceat(d, idx)

    minsr = np.repeat(mins, lens)
    maxsr = np.repeat(maxs, lens)

    D = max_val - min_val
    scaled_01_vals = (d - minsr) / (maxsr - minsr)
    # d[:] = scaled_01_vals * D + min_val
    res_data[:] = scaled_01_vals * D + min_val
    return res



pointer_annot_upd = []
pointer_annot = []
scs = []


def plot_scatter_names(ax, fig, color_map, x, y, values, labels, title, x_title, y_title):
    norm = plt.Normalize()
    cmap = color_map

    sc = ax.scatter(x, y, s=0.25, c=values, cmap=cmap, norm=norm, alpha=0.4)
    scs.append(sc)

    # for i, label in enumerate(labels):
    #     ax.annotate(label, (x[i], y[i]), fontsize=4)

    ax.set(xlabel=x_title, ylabel=y_title)
    ax.legend(*sc.legend_elements(num=5))
    ax.set_facecolor('black')

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->", color="white"))

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(":".join(list(map(str, ind["ind"]))),
                               ":".join([labels[n] for n in ind["ind"]]))
        print("hov:", text)
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(values[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)

    pointer_annot.append(annot)
    pointer_annot_upd.append(update_annot)


def train_test_lr(data, target, test_frac):
    classifiers = {"LR": LogisticRegression(),
                   "SGD": SGDClassifier(),
                   # "GNB": GaussianNB(),
                   # "MNB": MultinomialNB()
                   }

    X_train, X_test, y_train, y_test = train_test_split(data, target,
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


print("Calculating statistics.")
# non_zero_norm_word_score = non_zero_norm(word_scores_by_doc)
non_zero_mean_word_score = non_zero_mean(word_scores_by_doc)
non_zero_var_word_score = non_zero_variance(word_scores_by_doc)
non_zero_stdev_word_score = non_zero_stdev(word_scores_by_doc)
max_idf = np.max(vectorizer.idf_)

# print("NZ_NORM:", non_zero_norm_word_score.shape, np.min(non_zero_norm_word_score),  np.max(non_zero_norm_word_score))
print("NZ_MEAN:", non_zero_mean_word_score.shape, np.min(non_zero_mean_word_score),  np.max(non_zero_mean_word_score))
print("NZ_VAR:", non_zero_var_word_score.shape, np.min(non_zero_var_word_score), np.max(non_zero_var_word_score))
print("NZ_STDEV:", non_zero_stdev_word_score.shape, np.min(non_zero_stdev_word_score), np.max(non_zero_stdev_word_score))

#
# for feature_name, mean_score, stdev_score in sorted(zip(vectorizer.get_feature_names(),
#                                            non_zero_mean_word_score,
#                                            non_zero_stdev_word_score), key=lambda z: z[1]):
#     print(feature_name, mean_score, stdev_score)

print("Training models:")

print("TFIDF model:")
train_test_lr(tfidf_data, target, test_fraction)

print("Word scores model:")
train_test_lr(word_scores_by_doc, target, test_fraction)

print("Plotting.")
fig, axs = plt.subplots(nrows=1, ncols=2)
plt.setp(axs, adjustable='box')

plot_scatter_names(axs[0], fig, plt.cm.cool,
                   non_zero_mean_word_score,
                   non_zero_stdev_word_score,
                   vectorizer.idf_,
                   vectorizer.get_feature_names(),
                   "mean / stdev", "mean_score", "stdev_score")


plot_scatter_names(axs[1], fig, plt.cm.RdYlGn,
                   vectorizer.idf_,
                   non_zero_stdev_word_score,
                   non_zero_mean_word_score,
                   vectorizer.get_feature_names(),
                   "norm mean / stdev", "idf", "stdev_score")


def hover(event):
    cont_any_sc = False
    ind_any_sc = None
    for i in range(len(scs)):
        cont, ind = scs[i].contains(event)
        if cont:
            cont_any_sc = True
            ind_any_sc = ind
            break

    for i in range(len(scs)):
        annot = pointer_annot[i]
        update_annot = pointer_annot_upd[i]
        vis = annot.get_visible()
        if cont_any_sc:
            update_annot(ind_any_sc)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)


plt.show()
# mean_word_score = word_scores_by_doc.mean(axis=1)
# for feature_name, mean_score in sorted(zip(vectorizer.get_feature_names(), mean_word_score), key=lambda z: z[1]):
#     print("MEAN:", feature_name, mean_score)
