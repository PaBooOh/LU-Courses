
import pandas as pd
import numpy as np
from math import isnan
import joblib
import numpy.ma as ma
import threading
import multiprocessing
from multiprocessing import Manager
import random

# Multi-threads
class Threads(threading.Thread):
    def __init__(self, threadID, train_set, test_set, lr=0.005, k_factors=10, reg=0.05, epoch=75):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.train_set = train_set
        self.test_set = test_set
        self.lr = lr
        self.k_factors = k_factors
        self.reg = reg
        self.epoch = epoch
    
    def run(self):
        model = SVD(
            self.threadID, 
            self.train_set, 
            self.test_set, 
            lr=self.lr,
            K_features=self.k_factors,
            reg=self.reg,
            epoch=self.epoch)
        model.training(retrain=False)

class CrossValidation():
    def __init__(self, file_path, shuffle=True):
        self.file_path = file_path
        self.shuffle = shuffle

    def load_dataset(self, train_set_path, test_set_path):
        return joblib.load(train_set_path), joblib.load(test_set_path)

    def process_dataset(self):
        ratings = pd.read_csv(
            self.file_path,
            sep='::',
            engine='python',
            header=None,
            names=['UserID','MovieID','Rating','Timestamp'])

        ratings = ratings.pivot(
            index = 'UserID', 
            columns ='MovieID', 
            values = 'Rating')
        return ratings
    
    def split(self, fold=5, all=False, save=False):

        # process raw data in dataframe format
        ratings = self.process_dataset()

        # not folding but regarding whole dataset as train set just for test
        if all:
            return ratings.to_numpy()
        
        # use dictionary to save userId and its coordinatates of rating
        dict_ratings_valid = dict()
        for i, row in ratings.iterrows():
            u_rating = row[np.isnan(row) == False]
            u_rating_j = list(u_rating.index)
            random.seed(88)
            random.shuffle(u_rating_j)
            dict_ratings_valid[i] = u_rating_j
        
        # read dict and split data set into 5 parts and correspondingly generate 5 folds
        results = []
        for num in range(fold):
            # use dictionary to save userId and its coordinatates of rating
            dict_ratings_valid_train = dict()
            dict_ratings_valid_test = dict()
            train_set = ratings.copy()
            test_set = ratings.copy()
            fold_idx = [i for i in range(fold)]
            fold_idx.remove(num)
            for user_id in dict_ratings_valid.keys():
                tmp = [dict_ratings_valid[user_id][i::fold] for i in range(fold)]
                dict_ratings_valid_test[user_id] = tmp[num] # test_dict
                tr_tmp = []
                for idx in fold_idx:
                    tr_tmp.extend(tmp[idx]) 
                dict_ratings_valid_train[user_id] = tr_tmp # train_dict

            # get train set
            for user_id in dict_ratings_valid_test:
                cols = dict_ratings_valid_test[user_id]
                train_set.loc[user_id, cols] = np.nan
            
            # get test set
            for user_id in dict_ratings_valid_train:
                cols = dict_ratings_valid_train[user_id]
                test_set.loc[user_id, cols] = np.nan

            if save:
                joblib.dump(test_set.to_numpy(), 'test_set_' + str(num))
                joblib.dump(train_set.to_numpy(), 'train_set_' + str(num))
                print('Fold_' + str(num) + ' Saved')
            
            results.append([train_set.to_numpy(), test_set.to_numpy()])
        # results->    [x]    [y]    [z]
        # format->  |k-fold|tr/tst|user_id|
        return results

class SVD():
    def __init__(self, processID, train_set, test_set, K_features=10, lr=0.005, reg=0.05, epoch=75):
        self.processID =  processID
        self.train_set = train_set
        self.test_set = test_set
        self.K_features = K_features
        self.lr = lr
        self.reg = reg
        self.epoch = epoch
        self.train_results = []
        self.test_results = []
        
    # initialize params
    def initialize_latent_vectors(self):
        # To be reproducible
        np.random.seed(33)
        self.U = np.random.rand(self.train_set.shape[0], self.K_features)
        np.random.seed(33)
        self.M = np.random.rand(self.K_features, self.train_set.shape[1])
    
    # a) loss function
    def MAE(self, dataset):
        mask = np.isnan(dataset)
        masked_array = ma.array(dataset, mask=mask)
        error = masked_array - np.matmul(self.U, self.M)
        MAE = np.mean(np.absolute(error))
        return MAE

    # b) loss function
    def RMSE(self, dataset):
        mask = np.isnan(dataset)
        masked_array = ma.array(dataset, mask=mask)
        error = masked_array - np.matmul(self.U, self.M)
        RMSE = np.sqrt((error**2).mean())
        return RMSE
    
    def update_params(self, i, j, error):
        self.U[i,:] += self.lr * (2 * error * self.M[:,j] - self.reg * self.U[i,:])
        self.M[:,j] += self.lr * (2 * error * self.U[i,:] - self.reg * self.M[:,j])
    
    def save_params(self):
        joblib.dump(self.U, 'U_vec'+str(self.processID))
        joblib.dump(self.M, 'M_vec'+str(self.processID))
    
    def load_params(self):
        self.U = joblib.load('U_vec'+str(self.processID))
        self.M = joblib.load('M_vec'+str(self.processID))

    def observe(self, epoch, type):
        if type == 'test':
            RMSE = self.RMSE(self.test_set)
            MAE = self.MAE(self.test_set)
            print(self.processID,' Test-->','Epoch: ', epoch, 'RMSE: ', RMSE, 'MAE: ', MAE)
            self.test_results.append([RMSE,MAE])
            if epoch == self.epoch - 1:
                joblib.dump(self.test_results, str(self.processID)+'_testing_results')
        elif type == 'train':
            RMSE = self.RMSE(self.train_set)
            MAE = self.MAE(self.train_set)
            print(self.processID,' Train-->','Epoch: ', epoch, 'RMSE: ', RMSE, 'MAE: ', MAE)
            self.train_results.append([RMSE,MAE])
            if epoch == self.epoch - 1:
                joblib.dump(self.train_results, str(self.processID)+'_training_results')

    # training
    def training(self, retrain=False):
        # read local params files saved
        if retrain:
            self.load_params()
            print('Thread-->',self.processID,' Params loaded')
        else:
        # 1) or initialize params and train from scratch
            self.initialize_latent_vectors()
        # 2) start training
        '''
        a) Iterate over each known element
        b) Update the ith row of U and the jth column of M
        '''
        for epoch in range(self.epoch):
            for index, el in np.ndenumerate(self.train_set):
                if isnan(el) == False:
                    i, j = index
                    pred = np.dot(self.U[i,:], self.M[:, j]) # row vec * col vec  
                    error = el - pred # get error
                    self.update_params(i,j, error)
            # observe tranining process
            self.observe(epoch,'train')
            # get performance of model on test set
            self.observe(epoch,'test')
            # save params
            if epoch == self.epoch - 1:
                self.save_params()

def run_multi_process(train_set, test_set, lr=0.005, k_factors=10, reg=0.05, epoch=75):
    processID = multiprocessing.current_process().name
    model = SVD(
        processID, 
        train_set, 
        test_set, 
        lr=lr,
        K_features=k_factors,
        reg=reg,
        epoch=epoch)
    model.training(retrain=False)



if __name__ == "__main__":
    
    # 1) split and get 5 dataset including train&test
    cv = CrossValidation('ratings.dat')
    result = cv.split(fold=5) 
    
    # 2) call mutil-process and start training
    for num in range(5):
        train_set = result[num][0]
        test_set = result[num][1]
        process = multiprocessing.Process(
            target=run_multi_process,
            args=(train_set, test_set))
        process.start()

    # for num in range(5):
    #     train_set = result[num][0]
    #     test_set = result[num][1]
    #     thread = Threads(num, train_set, test_set, lr=0.005)
    #     thread.start()