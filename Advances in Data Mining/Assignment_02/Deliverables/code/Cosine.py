import numpy as np

# 0) define hyperparameters
hyper_planes_num = 100 # here we generate vectors that normal to these hyperplane instead.
bands_num = 6
rows_num = 15

# 1) Read the data and convert into sparse matrix (usr * mov)
def sparse_pivot(file_path):
    raw_ratings_data = np.load(file_path).astype('int32')
    user_idx, movie_idx, ratings = raw_ratings_data[:, 0]-1, raw_ratings_data[:, 1]-1, raw_ratings_data[:, 2]
    from scipy.sparse import csr_matrix
    ratings_mat = csr_matrix((ratings, (user_idx, movie_idx)))
    return ratings_mat

# 2) Generate signature matrix (dimention reduction) using random projection
class CosineSig():

    def __init__(self, seed, file_path):
        self.dataset = sparse_pivot(file_path)
        self.seed = seed
    
    def get_normal_vectors(self):
        '''
        Instead of generating complex hyperplanes, we generate vectors that is normal to hyperplanes generated randomly.
        So multiplying a normal vector by a vector in the original data matrix, we get the position that points are on which side of a hyperplane.
        '''
        movies_dim = self.dataset.shape[1]
        np.random.seed(self.seed)
        self.normal_vectors = np.random.randn(movies_dim, hyper_planes_num)
    
    def get_signatures_mat(self):
        '''
        the values/elements in the signature matrix are binary.
        '''
        # randomly pick some normal vectors
        self.get_normal_vectors()
        self.cosine_signatures = (self.dataset.dot(self.normal_vectors) >= 0).astype('int')
        return self.cosine_signatures
    
    def get_dataset(self):
        return self.dataset

# 3) Application of LSH and implementation of writing to local file
class CosineLSH():
    def __init__(self, seed, file_path):
        self.seed = seed
        self.file_path = file_path
    
    # split signature into b parts
    def banding(self):
        '''
        By class CosineSig, we convert the original data into signature matrix, which reduces the size of dimension.
        Next, due to the shape of this signature matrix is like (usr, bit), we partition this matrix by columns.
        If b*r is not exactly the same as the length of the column of sig mat, we just ignore the last part that differs with other parts.
        '''
        cs = CosineSig(self.seed, self.file_path)
        self.dataset = cs.get_dataset().toarray()
        # self.dataset = cs.get_dataset()
        cosine_signatures = cs.get_signatures_mat() # shape: (103703, 100)
        bands = []
        for i in range(0, cosine_signatures.shape[1], rows_num):
            bands.append(cosine_signatures[:,i:i+rows_num])
        
        return bands
    
    # assign similar pairs to same buckets
    def hash2buckets(self):
        '''
        After banding strategy, a hash function is applied to clustering similar users.
        For instance, signature vector of user 1 and 2 are [0,1,0,1,1,0] both, and they are definitely a candidate pair.
        So, we create a dictionary where a key could be a string 010110 and its value is a list consists the userID 1 and 2
        in str format.
        '''

        bands = self.banding()
        bands_lst = []
        for i in range(bands_num):
            tmp = []
            band_dict = dict()
            for usr_idx, user_sigs in enumerate(bands[i]):
                string = ''.join([str(x) for x in list(user_sigs)])
                if string in band_dict:
                    tmp = band_dict[string]
                tmp.append(usr_idx)
                band_dict.update({string:tmp})
                tmp = []
            bands_lst.append(band_dict)
            band_dict = dict()

        return bands_lst
    
    # find the 'real similar pairs' and write them to local file.
    def compute_similarity2file(self, cos_type):
        '''
        N.B.
        1) this method is to compute similarity for each candiate pair in buckets from different band and then
        write to .txt file pair-by-pair.
        2) threshold = 0.73 for both cosine and discrete cosine
        3) for discrete cosine, we convert user vectors into binary format e.g., [5,0,1] - > [1,0,1]
        4) To avoid duplicate pairs existing .txt file at the same time, we store user A and B like 'userA@userB'.
        e.g., user 47930 and 53615 are a candidate over given threshold, we join them like '47930@53615' to keep unique and
        check each time when writing to .txt file.
        5) indices for user-movie matrix range from 0, but we restore it to the userId format when writing to .txt file.
        '''
        from math import acos
        from itertools import combinations # compare two elements in a list
        assert cos_type in ['cs', 'dcs']
        file_path = 'cs.txt' if cos_type == 'cs' else 'dcs.txt'
        bands_lst = self.hash2buckets() # consists of series steps and finally get buckets containing at least 1 user.
        pairs = set() # To write unique pairs to txt file, use set() to avoid duplicate.
        print('>>>>>>Start writing to ' + file_path)
        for i in range(bands_num):
            for key in bands_lst[i]:
                candidate_pairs = bands_lst[i][key]
                if len(candidate_pairs) >= 2:
                    for userA, userB in combinations(candidate_pairs, 2):
                        # in case of duplicate pairs
                        pair_str = str(userA+1) + '@' + str(userB+1) 
                        if pair_str in pairs:
                            continue
                        pairs.add(pair_str) # store all uniqute candidate pairs
                        # get UserVectors by UserId
                        vec1, vec2 = self.dataset[userA], self.dataset[userB]
                        # vec1 = self.dataset.indices[self.dataset.indptr[userA]:self.dataset.indptr[userA+1]]
                        # vec2 = self.dataset.indices[self.dataset.indptr[userB]:self.dataset.indptr[userB+1]]

                        # *only for computing sim of discrete cosine (this is the only difference between discrete cosine and cosine)
                        if cos_type == 'dcs': 
                            vec1 = (vec1 > 0).astype('int') # ratings(1-5) -> all 1
                            vec2 = (vec2 > 0).astype('int')
                        # compute cosine similarity
                        cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        rad = acos(cos)
                        deg = np.rad2deg(rad)
                        sim = 1 - deg/180
                        sim = round(sim, 4)
                        # set threshold to 0.73
                        if sim > 0.73:
                            # write pair per line to cs.txt
                            with open(file_path,"a") as f:
                                f.write(str(userA+1) + ',' + str(userB+1) + '\n')
                            print('Band: ', i ,'Real Pair: ' , userA+1, userB+1)
        print('>>>>>>All Done')
    

# Tool class for several similarity computation (personal use only, not for the main process)
class CosineSim():

    # get the cosine similarity between two vectors
    @staticmethod
    def get_cosine_similarity(sim1, sim2):
        dot_product = np.dot(sim1, sim2)
        norm_product = np.linalg.norm(sim1) * np.linalg.norm(sim2)
        cos_theta = dot_product / norm_product
        return round(cos_theta, 4)

    @staticmethod
    def get_cosine_theta(sim1, sim2):
        from math import acos
        cos_theta = CosineSim.get_cosine_similarity(sim1, sim2)
        rad = acos(cos_theta)
        theta = np.rad2deg(rad)
        return round(theta, 2)
    
    @staticmethod
    def get_cosine_similarity_degree(sim1, sim2):
        theta = CosineSim.get_cosine_theta(sim1, sim2)
        deg_sim = 1 - theta/180
        return round(deg_sim, 4)
    
    @staticmethod
    def compute_vecs_deg_sim(idx1, idx2):
        dataset = sparse_pivot().toarray()
        v_usr1, v_usr2 = dataset[idx1], dataset[idx2]
        sim = CosineSim.get_cosine_theta(v_usr1, v_usr2)
        print(sim)
    
    @staticmethod
    def compute_vecs_sim(idx1, idx2, type='original'):
        assert type in ['original', 'signature']
        if type == 'original':
            dataset = sparse_pivot('user_movie_rating.npy').toarray()
        elif type == 'signature':
            run = CosineSig(123)
            dataset = run.get_signatures_mat()
        v_usr1, v_usr2 = dataset[idx1], dataset[idx2]
        sim = CosineSim.get_cosine_similarity_degree(v_usr1, v_usr2)
        print(sim)


# if __name__ == "__main__": 
# # Just for test
    # CosineSim.compute_vecs_sim(8294,59975, type='original')
