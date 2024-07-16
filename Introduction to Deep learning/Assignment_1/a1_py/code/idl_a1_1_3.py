
import pandas as pd
from idl_a1_1_1 import simple_classifier

# extract dataset
train_set = pd.read_csv('./datasets/train_in.csv',header=None).values
train_label = pd.read_csv('./datasets/train_out.csv',header=None).values
test_set = pd.read_csv('./datasets/test_in.csv',header=None).values
test_label = pd.read_csv('./datasets/test_out.csv',header=None).values

if __name__ == "__main__":  
    agg_accuracy, digit_correct_ratio, digit_correct_num, digit_num, _ = simple_classifier(train_set,train_label,test_set,test_label,pca_coeff=0.9)
    ''' *Test: digits distribution
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