import pandas as pd

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold

train_data = pd.read_csv("/home/math-tr/lsh/Ariel/data/train_in.csv").values  
train_label = pd.read_csv("/home/math-tr/lsh/Ariel/data/train_out.csv").values
test_data = pd.read_csv("/home/math-tr/lsh/Ariel/data/test_in.csv").values  
test_label = pd.read_csv("/home/math-tr/lsh/Ariel/data/test_out.csv").values

train_data_KMeans = KMeans(n_clusters=10).fit_transform(train_data)
train_data_PCA = decomposition.PCA(n_components=2).fit_transform(train_data)
train_data_LLE = LocallyLinearEmbedding(n_components=2).fit_transform(train_data)
train_data_tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, n_iter=1000, verbose=1).fit_transform(train_data)
train_data_KNN = KNeighborsClassifier(n_neighbors=10).fit_transform(train_data)
'''
model = KNeighborsClassifier(n_neighbors=10)
model.fit(train_data,train_label)
train_score = model.score(train_data, train_label)
test_score = model.score(test_data, test_label)
print("train accuracy=%.2f%%" % ( train_score * 100))
print("test accuracy=%.2f%%" % ( test_score * 100))
'''
