
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd


# Q01. able to addresses all 20 categories
twenty_train = fetch_20newsgroups(data_home=None, subset='train', categories=None, shuffle=True, random_state=123)
twenty_test = fetch_20newsgroups(data_home=None, subset='test', categories=None, shuffle=True, random_state=321)

# print("\n".join(twenty_train.data[0].split("\n")[:3]))
count_vect = CountVectorizer()
X_train_tc = count_vect.fit_transform(twenty_train.data)

tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tc)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_tc)
X_train_tf = tf_transformer.transform(X_train_tc)

print(X_train_tf != X_train_tfidf)


