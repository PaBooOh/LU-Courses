import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from idl_a1_1_1 import simple_classifier

# extract dataset
train_set = pd.read_csv('./datasets/train_in.csv',header=None).values
train_label = pd.read_csv('./datasets/train_out.csv',header=None).values
test_set = pd.read_csv('./datasets/test_in.csv',header=None).values
test_label = pd.read_csv('./datasets/test_out.csv',header=None).values

# PCA
pca = PCA(0.8)
pca.fit(train_set)
train_set_reduction = pca.transform(train_set)
test_set_reduction = pca.transform(test_set)
datasets_X = {'on test sets': [test_set_reduction, test_label], 'on train sets': [train_set_reduction, train_label]}

# create KNN
knn_model = KNN(n_neighbors=6, algorithm='auto', weights='distance', n_jobs=1)
knn_model.fit(train_set_reduction, np.ravel(train_label))


# KNN on data sets
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey='row')
for i in range(2):
    for index, (desc,dataset_X) in enumerate(datasets_X.items()):
        pred = knn_model.predict(dataset_X[0]) if i == 0 else simple_classifier(train_set_reduction, train_label, dataset_X[0], dataset_X[1], pca_coeff='None')[-1]
        title_label = 'KNN ' if i == 0 else 'Distance-based '
        cm = confusion_matrix(dataset_X[1], pred, labels=knn_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
        disp.plot(ax=axes[index+2*i], xticks_rotation=45)
        disp.ax_.set_title(title_label + desc)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if index != 0 or i != 0:
            disp.ax_.set_ylabel('')

fig.text(0.45, 0.1, 'Predicted label')
plt.subplots_adjust(wspace=0.2, hspace=0.1)
fig.colorbar(disp.im_, ax=axes, fraction=0.008)
plt.show()


# if __name__ == "__main__":  
#     print()