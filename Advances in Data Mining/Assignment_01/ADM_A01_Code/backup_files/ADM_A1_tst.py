# cv for User/Movie matrix only
class CrossValidation():
    def __init__(self, file_path, shuffle=True, normolize=False):
        self.file_path = file_path
        self.normolize = normolize
        self.shuffle = shuffle

    def process_dataset(self):
        ratings = pd.read_csv(
            self.file_path,
            sep='::',
            engine='python',
            header=None,
            names=['UserID','MovieID','Rating','Timestamp'])
        ratings = ratings.drop(['Timestamp'], axis=1)
        ratings = ratings.values
        
        # groupby
        group = np.split(ratings, np.unique(ratings[:, 0], return_index=True)[1][1:])
        if self.shuffle:
            for user in group:
                np.random.seed(88)
                np.random.shuffle(user)
        return group, ratings
        
    def split(self, fold=5, all=False, save=False):
        # process raw data
        group, r = self.process_dataset()

        # not folding but regarding whole dataset as train set
        if all:
            return r

        # folding
        users = []
        for user in group:
            folds = np.array_split(user, fold)
            users.append(folds)
        users = np.array(users)
        
        results = []
        for i in range(fold):
            train_set = []
            test_set = []
            for user in users:
                tr = np.delete(user, i)
                train_tmp = np.vstack(tr)
                test_tmp = user[i]
                train_set.append(train_tmp)
                test_set.append(test_tmp)
            train_set = np.vstack(train_set)
            test_set = np.vstack(test_set)
            train_set = 
            results.append([train_set, test_set])
            # print(train_set.shape, test_set.shape)
        return results

        
        # userid_group = ratings.groupby('UserID')
        # for index, (key, value) in enumerate(userid_group):
        #     # num = round(alpha*value.shape[0])
        #     sample = value.sample(frac=alpha, replace=False, random_state=88)
        #     # initialize
        #     if index == 0:
        #         train_set = sample
        #         continue
        #     train_set = train_set.append(sample)
        
        # # get test
        # tr_idx = train_set.index.to_list()
        # test_set = ratings[~ratings.index.isin(tr_idx)]

        # # pivot
        # train_set = train_set.pivot(
        #     index = 'UserID', 
        #     columns ='MovieID', 
        #     values = 'Rating')
        # # train_set = train_set.to_numpy() # to ndarray

        # test_set = test_set.pivot(
        #     index = 'UserID', 
        #     columns ='MovieID', 
        #     values = 'Rating')
        # # test_set = test_set.to_numpy() # to ndarray
        
        # normolize
        # if self.normolize:
        #     ratings_mean_tr = np.nanmean(train_set, axis=1)
        #     ratings_mean_tt = np.nanmean(test_set, axis=1)
        #     train_set = train_set - ratings_mean_tr.reshape(-1, 1)
        #     test_set = test_set - ratings_mean_tt.reshape(-1, 1)
        
        # if save:
        #     joblib.dump(train_set, train)
        #     joblib.dump(test_set, 'U_vec'+str(self.threadID))