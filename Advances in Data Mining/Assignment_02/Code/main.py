import sys, getopt
from Cosine import CosineLSH # cosine & discrete cosine
from Jaccard import Jaccard_s # jaccard

cmd_arg = sys.argv[1:]
# -d /very/long/path/to/data.npy -s 2021 -m dcs
assert cmd_arg[0] in ['-d', '-s', '-m'] # '-s'
assert cmd_arg[2] in ['-d', '-s', '-m'] # '-s'
assert cmd_arg[4] in ['-d', '-s', '-m'] # '-m'

file_path = cmd_arg[cmd_arg.index('-d') + 1] # str
seed = int(cmd_arg[cmd_arg.index('-s') + 1]) # int
similarity_measure = cmd_arg[cmd_arg.index('-m') + 1] # str -> js/cs/dcs
assert similarity_measure in ['js', 'cs', 'dcs']
# print(file_path, seed, similarity_measure)

if similarity_measure in ['cs', 'dcs']:
    run = CosineLSH(seed, file_path)
    run.compute_similarity2file(similarity_measure)

elif similarity_measure == 'js':
    Jaccard_s(seed, file_path)


