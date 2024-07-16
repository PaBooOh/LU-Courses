from inspect import CO_ITERABLE_COROUTINE
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''
implement distance-based classifier using notion of center of points
'''

# extract dataset
train_set = pd.read_csv('./datasets/train_in.csv',header=None).values
train_label = pd.read_csv('./datasets/train_out.csv',header=None).values
test_set = pd.read_csv('./datasets/test_in.csv',header=None).values
test_label = pd.read_csv('./datasets/test_out.csv',header=None).values
size_train_set, ndim_train_set = train_set.shape
size_test_set, ndim_test_set = test_set.shape

# sort from lowest to largest
def sort_mnist(data_X, data_Y):
    label_index = np.argsort(data_Y.flatten())
    X_sorted = data_X[label_index]
    Y_sorted = data_Y[label_index].flatten().tolist()
    return X_sorted, Y_sorted

# calculate 10 centers of digits
def cal_digits_center(train_X, train_Y):

    # sort in order(low to hi) and return index : classify mnist by digit/label
    train_set_sorted, train_label_sorted = sort_mnist(train_X, train_Y)

    # sort and calculate points of center of each digits
    digit_bound = 0
    lst_digit_center = []
    for digit in range(10):
        temp = digit_bound
        digit_bound = len(train_label_sorted) - train_label_sorted[::-1].index(digit)
        lst_digit_center.append(np.sum(train_set_sorted[temp:digit_bound], axis=0)/digit_bound)
    return lst_digit_center

def cal_center_distance(train_X, train_Y, digit_1, digit_2):
    if digit_1 == digit_2:
        return 0
    lst_digit_center = cal_digits_center(train_X, train_Y)
    return distance.euclidean(lst_digit_center[digit_1], lst_digit_center[digit_2])

# q1.1/q1.3
def simple_classifier(train_X,train_Y,test_X,test_Y,pca_coeff):

    # PCA on train/test
    if isinstance(pca_coeff,float) and pca_coeff < 1:
        pca = PCA(pca_coeff)
        pca.fit(train_X)
        train_X = pca.transform(train_X)
        test_X = pca.transform(test_X)
    
    # calculate center of cluster of digits
    lst_digit_center = cal_digits_center(train_X, train_Y)

    # comparison, obtain min distance that can determine what kind of digit to which input belong
    agg_correct_num = 0
    digit_correct_num = [0 for _ in range(10)]
    digit_num = [0 for _ in range(10)]
    test_Y = test_Y.flatten()
    predictions = []
    for i in range(len(test_Y)):
        dst = [distance.euclidean(test_X[i],lst_digit_center[digit]) for digit in range(10)]
        pred = dst.index(min(dst))
        predictions.append(pred)
        digit_num[test_Y[i]] += 1
        if pred == test_Y[i]:
            digit_correct_num[pred] += 1
            agg_correct_num += 1
    digit_correct_ratio = [digit_correct_num[i]/digit_num[i] for i in range(len(digit_num))]
    agg_accuracy = agg_correct_num/len(test_Y)
    return agg_accuracy, digit_correct_ratio, digit_correct_num, digit_num, predictions

def plot_bar(corr_num, total_num, lst_acc, title):
    plt.figure(figsize=(12,7))
    x = [str(i) for i in range(10)]
    plt.bar(x=x, height=corr_num, color='red', width=0.2, label='Digits predicted correctly')
    plt.bar(x=x, height=total_num, color='black', align='edge', width=0.2, label='Total Digits')
    for index, (x, total) in enumerate(zip(x, total_num)): 
        plt.text(x, total, 'acc = ' + '%.f' % (lst_acc[index] * 100)+'%', ha='center', va= 'bottom', fontsize=11, color='green') 
    plt.xlabel('Digits')
    plt.ylabel('Respective Numbers of digits and digits predicted correctly')
    plt.legend()
    plt.title(title)
    plt.show()

if __name__ == "__main__": 

    # (1) show the performance of distance-based classifier on test set
    agg_accuracy, digit_correct_ratio, digit_correct_num, digit_num, _ = simple_classifier(train_set,train_label,test_set,test_label,pca_coeff=0.9)
    digits_accuracy = np.divide(digit_correct_num, digit_num)
    plot_bar(digit_correct_num, digit_num, digits_accuracy, 'Test')

    # (2) show the performance of distance-based classifier on train set
    agg_accuracy, digit_correct_ratio, digit_correct_num, digit_num, _ = simple_classifier(train_set,train_label,train_set,train_label,pca_coeff=0.9)
    digits_accuracy = np.divide(digit_correct_num, digit_num)
    plot_bar(digit_correct_num, digit_num, digits_accuracy, 'Train')
    # print(agg_accuracy)


    # (3) distij = dist(ci, cj ) , calculate the distances between the centers of the 10 clouds.

    print('Between-class distance matrix: ')
    dict_dist_col_ij = dict()
    for i in range(10):
        lst_dist_col_ij = []
        for j in range(10):
            centers_distance = cal_center_distance(train_set, train_label, i, j)
            lst_dist_col_ij.append(centers_distance)
        dict_dist_col_ij[str(i)] = lst_dist_col_ij
    
    df = pd.DataFrame(dict_dist_col_ij)
        

    print(df)


    ''' Results

        *Test: digits distribution
        Overall: [224, 121, 101, 79, 86, 55, 90, 64, 92, 88]
        Correct: [186, 121, 69, 68, 66, 15, 66, 54, 53, 30]
        digits_correct_rate: [0.8303571428571429, 1.0, 0.6831683168316832, 0.8607594936708861, 0.7674418604651163, 0.2727272727272727, 0.7333333333333333, 0.84375, 0.5760869565217391, 0.3409090909090909]
        Accuracy: 0.728
        ________________________________________________________________________________________________________
        ________________________________________________________________________________________________________
        *Train: digits distribution
        Overall: [319, 252, 202, 131, 122, 88, 151, 166, 144, 132]
        Correct: [270, 252, 172, 121, 95, 30, 109, 155, 90, 38]
        digits_correct_rate: [0.8463949843260188, 1.0, 0.8514851485148515, 0.9236641221374046, 0.7786885245901639, 0.3409090909090909, 0.7218543046357616, 0.9337349397590361, 0.625, 0.2878787878787879]
        Accuracy: 0.7803163444639719
    '''
