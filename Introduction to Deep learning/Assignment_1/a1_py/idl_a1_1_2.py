
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from idl_a1_1_1 import sort_mnist


# extract dataset
train_set = pd.read_csv('datasets/train_in.csv',header=None).values
train_label = pd.read_csv('datasets/train_out.csv',header=None).values
test_set = pd.read_csv('datasets/test_in.csv',header=None).values
test_label = pd.read_csv('datasets/test_out.csv',header=None).values
size_train_set, ndim_train_set = train_set.shape
size_test_set, ndim_test_set = test_set.shape




def classify_mnist(train_X, train_Y):

    # classify
    digit_bound = 0
    digit_classified = []
    for digit in range(10):
            temp = digit_bound
            digit_bound = len(train_Y) - train_Y[::-1].index(digit)
            digit_classified.append(train_X[temp:digit_bound])
    return digit_classified

def pca_visualization_3d(train_X, train_Y):

    # pca: dimentionality reduction on train
    pca = PCA(n_components=3)
    pca.fit(train_X)
    train_X = pca.transform(train_X)
    # sort and classify
    train_set_sorted, train_label_sorted = sort_mnist(train_X, train_Y)
    digit_classified = classify_mnist(train_set_sorted, train_label_sorted)
    # PCA visualization
    cls_label = np.unique(train_label_sorted)
    markers = ['s','x','o','.',',','<','>', '^','8','*']
    colors = list(plt.rcParams['axes.prop_cycle'])
    ax = plt.figure(figsize=(12,6)).gca(projection='3d')
    for i, (y, m) in enumerate(zip(cls_label, markers)):
            digit_cluster = digit_classified[i]
            ax.scatter(xs=digit_cluster[:, 0], ys=digit_cluster[:, 1], zs=digit_cluster[:, 2], c=colors[i]['color'], cmap='tab10',label = y, alpha=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('PCA')
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.show()

def tSNE_visualization_2d(train_X, train_Y):

    # tsne
    tsne = TSNE(n_components=2,init='pca', random_state=321)
    train_X = tsne.fit_transform(train_X)

    # sort and classify
    train_set_sorted, train_label_sorted = sort_mnist(train_X, train_Y)
    digit_classified = classify_mnist(train_set_sorted, train_label_sorted)

    # t-SNE visualization
    cls_label = np.unique(train_label_sorted)
    markers = ['s','x','o','.',',','<','>', '^','8','*']
    colors = list(plt.rcParams['axes.prop_cycle'])
    plt.figure(figsize=(12,6))
    for i, (y, m) in enumerate(zip(cls_label, markers)):
            digit_cluster = digit_classified[i]
            plt.scatter(digit_cluster[:, 0], digit_cluster[:, 1], c=colors[i]['color'], cmap='tab10',label = y, marker=m, alpha=0.8)

    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('t-SNE')
    plt.show()

def LLE_visualization_2d(train_X, train_Y):

    # LLE
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors = 10, random_state=123)
    train_X = lle.fit_transform(train_X)

    # sort and classify
    train_set_sorted, train_label_sorted = sort_mnist(train_X, train_Y)
    digit_classified = classify_mnist(train_set_sorted, train_label_sorted)

    # LLE visualization
    cls_label = np.unique(train_label_sorted)
    markers = ['s','x','o','.',',','<','>', '^','8','*']
    colors = list(plt.rcParams['axes.prop_cycle'])
    plt.figure(figsize=(12,6))
    for i, (y, m) in enumerate(zip(cls_label, markers)):
            digit_cluster = digit_classified[i]
            plt.scatter(digit_cluster[:, 0], digit_cluster[:, 1], c=colors[i]['color'], cmap='tab10',label = y, alpha=0.5)

    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('LLE')
    plt.show()
    return

if __name__ == "__main__":  
    # pca_visualization_3d(train_set,train_label)
    # tSNE_visualization_2d(train_set, train_label)
    LLE_visualization_2d(train_set, train_label)