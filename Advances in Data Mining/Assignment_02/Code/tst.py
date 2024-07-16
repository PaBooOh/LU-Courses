import numpy as np
from math import acos

# 0) define hyperparameters
hyper_planes_num = 100 # here we generate vectors that normal to these hyperplane instead.
bands_num = 6
rows_num = 15

def sparse_pivot(file_path):
    raw_ratings_data = np.load(file_path)
    user_idx, movie_idx, ratings = raw_ratings_data[:, 0]-1, raw_ratings_data[:, 1]-1, raw_ratings_data[:, 2]
    from scipy.sparse import csr_matrix
    ratings_mat = csr_matrix((ratings, (user_idx, movie_idx)))
    return ratings_mat

def compare(a, b, file_path):
    dataset = sparse_pivot(file_path)
    a = dataset.indices[dataset.indptr[a]:dataset.indptr[a+1]]
    b = dataset.indices[dataset.indptr[b]:dataset.indptr[b+1]]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    rad = acos(cos)
    deg = np.rad2deg(rad)
    sim = 1 - deg/180
    sim = round(sim, 4)

ratings = sparse_pivot('user_movie_rating.npy')
# print(ratings)
print(ratings.indices[ratings.indptr[0]:ratings.indptr[1]])
# print(ratings.indices[ratings.indptr[0]:ratings.indptr[1]])
# compare(30250, 101403, 'user_movie_rating.npy')