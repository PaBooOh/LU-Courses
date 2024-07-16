
import pandas as pd
import numpy as np
from math import isnan
import joblib
import numpy.ma as ma
from sklearn.metrics import mean_squared_error as mse
import threading

# Multi-threads
class Threads(threading.Thread):
    def __init__(self, threadID, k_fold_dataset, lr=0.005):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.k_fold_dataset = k_fold_dataset
        self.lr = lr
    
    def run(self):
        train_set = self.k_fold_dataset[self.threadID-1][0]
        test_set = self.k_fold_dataset[self.threadID-1][1]
        model = SVD(
            self.threadID, 
            train_set, 
            test_set, 
            lr=self.lr)
        model.training()

# cv for User/Movie matrix only
class CrossValidation():
    def __init__(self, file_path, normolize=False):
        self.file_path = file_path
        self.normolize = normolize

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
        ratings = ratings.to_numpy() # to ndarray
        
        if self.normolize:
            ratings_mean = np.mean(ratings, axis=1)
            ratings = ratings - ratings_mean.reshape(-1, 1)
        return ratings
        
        
    def split(self, fold=5, shuffle=False, all=False):
        
        # process raw data
        dataset = self.process_dataset()
        # not fold but regard whole dataset as train set
        if all:
            return dataset
        # shuffle the sequence of data
        if shuffle:
            # To be reproducible
            np.random.seed(123)
            np.random.shuffle(dataset)
        # split dataset into k-folds(parts)  
        parts = np.array_split(dataset, fold)
        results = []
        # combine 4 out of 5 for training and leave 1 for testing
        for i in range(fold):
            train_idx = [idx for idx in range(fold)]
            train_idx.remove(i)
            train_set = np.vstack([*[parts[idx] for idx in train_idx]])
            # results: [[train_set1, test_set1],...]
            results.append([train_set, parts[i]])
        return results
        
class SVD():
    def __init__(self, threadID, train_set, test_set, K_features=10, lr=0.005, reg=0.05, epoch=75):
        self.threadID =  threadID
        self.train_set = train_set
        self.test_set = test_set
        self.K_features = K_features
        self.lr = lr
        self.reg = reg
        self.epoch = epoch
    
    # initialize params
    def initialize_latent_vectors(self):
        # To be reproducible
        np.random.seed(11)
        self.U = np.random.rand(self.train_set.shape[0], self.K_features)
        np.random.seed(44)
        self.M = np.random.rand(self.K_features, self.train_set.shape[1])
    
    # 1) loss function
    def MAE(self, dataset):
        mask = np.isnan(dataset)
        masked_array = ma.array(dataset, mask=mask)
        error = masked_array - np.matmul(self.U, self.M)
        MAE = np.mean(np.absolute(error))
        return MAE

    # 2) loss function
    def RMSE(self, dataset):
        mask = np.isnan(dataset)
        masked_array = ma.array(dataset, mask=mask)
        error = masked_array - np.matmul(self.U, self.M)
        RMSE = np.mean(error**2) ** 0.5
        return RMSE
    
    def update_params(self, i, j, error):
        self.U[i,:] += self.lr * (2 * error * self.M[:,j] - self.reg * self.U[i,:])
        self.M[:,j] += self.lr * (2 * error * self.U[i,:] - self.reg * self.M[:,j])
    
    def save_params(self):
        joblib.dump(self.U, 'U_vec'+str(self.threadID))
        joblib.dump(self.M, 'M_vec'+str(self.threadID))
    
    def load_params(self):
        self.U = joblib.load('U_vec'+str(self.threadID))
        self.M = joblib.load('M_vec'+str(self.threadID))

    def testing(self):
        MAE = self.MAE(self.test_set)
        RMSE = self.RMSE(self.test_set)
        print('Thread-->',self.threadID,' Test-->', 'RMSE: ', RMSE, 'MAE: ', MAE)
        
    
    # training
    def training(self, retrain=False):

        if retrain:
            self.load_params()
            print('Thread-->',self.threadID,' Params loaded: ')
        else:
        # initialize params
            self.initialize_latent_vectors()

        # start training
        valid_el = []
        for index, el in np.ndenumerate(self.train_set):
            if isnan(el) == False:
                valid_el.append([index, el])
        '''
        1) Iterate over each known element 
        2) update the ith row of U and the jth column of M.
        '''
        for epoch in range(self.epoch):
            print('Thread-->',self.threadID,' Epoch: ', epoch)
            # iterate over originial matrix, U&M matrix
            # obtain values and index without unknown element
            for index, rating_idx in enumerate(valid_el):
                i = rating_idx[0][0]
                j = rating_idx[0][1]
                rating = rating_idx[1]
                pred = np.dot(self.U[i,:], self.M[:, j]) # row vec * col vec  
                error = rating - pred # get error
                # update each param item
                self.update_params(i, j, error)
                if index % 10 == 0:
                    MAE = self.MAE(self.train_set)
                    RMSE = self.RMSE(self.train_set)
                    print(i,'Thread-->',self.threadID,' Train-->', 'RMSE: ', RMSE, 'MAE: ', MAE)
                # if index % 200 == 0:
                #     self.testing() # get performance of model on test set
                if index != 0 and index % 500 == 0:
                    self.save_params()
                    print('Thread-->',self.threadID,' Params saved')
        # self.testing() # get performance of model on test set
        

if __name__ == "__main__":
   
    cv = CrossValidation('ratings.dat')
    fold5_dataset = cv.split(fold=5, shuffle=True, all=True) # split and get 5 dataset including train&test
    model = SVD(
        1,
        fold5_dataset, 
        fold5_dataset,
        lr=0.01
        )
    model.training(retrain=False)
   

    # mutil-thread
    # for num in range(1):
    #     thread = Threads(num, fold5_dataset, lr=0.1)
    #     # thread.daemon = True
    #     thread.start()




    '''
    read_small

    # user_id, movie_id and ratings only
        # u_r_m = pd.read_csv(self.file_path, usecols=[0,1,2]).values
        # # to df
        # UM_ratings = pd.DataFrame(
        #     u_r_m, 
        #     columns=['UserID', 'MovieID', 'Rating'])
        # # ease the way to fit(U&M)
        # UM_ratings = UM_ratings.pivot(
        #     index = 'UserID', 
        #     columns ='MovieID', 
        #     values = 'Rating').fillna(0)

        # UM_ratings = UM_ratings.values # to ndarray
        # if self.normolize:
        #     ratings_mean = np.mean(UM_ratings, axis=1)
        #     UM_ratings = UM_ratings - ratings_mean.reshape(-1, 1)
        
        # return UM_ratings
    '''
       
    
    
    
