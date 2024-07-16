
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
# define hyperparameters
k_hash=320
r=8
b=k_hash
n_r=int(b/r)
dataset = None

# read data and convert into sparse matrix (usr * mov)
def Jaccard_s(seed,file_path):
    print('>>>>>>Start writing to ' + 'js.txt')
    data = np.load(file_path).astype('int32')
    global dataset
    data=csc_matrix((data[:,2],(data[:,1]-1,data[:,0]-1)))
    # data=csc_matrix((data[:,2],(data[:,1]-1,data[:,0]-1))).toarray()
    # data=data[1:,1:]
    # data=csc_matrix(data)
    dataset = data
    n_movie,n_user=data.shape
    index = np.arange(np.shape(data)[0])
    signature=np.zeros((n_user,k_hash))+99999
    for i in range(k_hash):
        np.random.seed(seed+i)
        np.random.shuffle(index)
        shuffle_data=data[index,:]
        for j in range(n_user):
            signature[j,i]=np.min(shuffle_data.indices[shuffle_data.indptr[j]:shuffle_data.indptr[j+1]])
    for i in range (n_r):
        locals()['d'+str(i)]={}
        keys=tuple(map(tuple,signature[:,i:i+r]))
        for j in range(n_user):     
            if keys[j] not in locals()['d'+str(i)]:
                locals()['d'+str(i)][keys[j]] = [j]
            else:
                locals()['d'+str(i)][keys[j]].append(j)
    similar_pairs=set()
    pairs=set()
    for n_dic in range(n_r):
        #print(n_dic)
        for value in locals()['d'+str(n_dic)].values():
            n=len(value)
            if n<20:
                for i in range(n):
                    for j in range(i+1,n):
                        if (value[i],value[j]) not in pairs:
                            pairs.add((value[i],value[j]))
                            similar=compare(value[i],value[j])
                            if similar>=0.5:
                                if (value[i],value[j]) not in similar_pairs:
                                    similar_pairs.add((value[i],value[j]))
                                    with open('js.txt',"a") as f:
                                        f.write(str(value[i]+1) + ',' + str(value[j]+1) + '\n')
                                    print('Real Pair: ' , str(value[i]+1), str(value[j]+1))
                                #print(similar_pairs)
                        else:
                            continue
    print('>>>>>>All Done')

def compare(a,b):
    a=dataset.indices[dataset.indptr[a]:dataset.indptr[a+1]]
    b=dataset.indices[dataset.indptr[b]:dataset.indptr[b+1]]
    return(len([val for val in a if val in b])/len(list(set(a).union(set(b)))))

# if __name__ == "__main__": 
#     Jaccard_s(123,'user_movie_rating.npy')
    # compare(18865, 19151)
    # Jaccard_s(2010,'user_movie_rating.npy')