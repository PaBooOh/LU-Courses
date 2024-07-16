
# bonus (hash function for string)
import ctypes

def hashFunction_DJB(str):
    hash = 5381
    # str_len = len(str)
    for char in str:
        hash = (((hash << 5) + hash) + ord(char))
    print(ctypes.c_uint32(hash).value) # < 2**32 (unsigned 32bit)

while(True):
    hashFunction_DJB(input())
