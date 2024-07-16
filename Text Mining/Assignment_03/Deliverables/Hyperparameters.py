from pickle import TRUE


SEED = 123 # for all functions that require random seed

# Dataset
DATASET_PATH = 'dataset/downloaded/twitter.tsv'
# DATASET_PATH = 'dataset/downloaded/twitter2016.tsv'
TEXT_SIZE = 0.25
# PERMUTATION = True

# Word2Vec
W2V_MODEL_FOLDER = 'model/w2v/'
W2V_SKIPGRAM_MODEL_FILE = 'model/w2v/w2v_dim300.ml'
WORD_DIM = 300
WINDOW_SIZE = 5
MIN_COUNT = 1
W2V_EPOCHS = 20
SG = 1

# Bag of words
MAX_VOCABS = 1000


# Naive Bayes
N_GRAM_IDF = (1, 3)
BINARY_IDF = True
SMOOTH_IDF = True

# LSTM training
TEXT_LEN = 20 # Keras embedding
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS= 12
TENSORBOARD_DIR = 'tensorboard/'
LSTM_SAVED_MODEL_PATH = 'model/lstm/'
LSTM_LOADED_MODEL_PATH = 'model/lstm/lstm_epoch4020220115-172659'

# BERT
MAX_LEN_BERT = 20
LR_BERT = 2e-5
BATCH_SIZE_BERT = 32
EPOCHS_BERT = 4
TENSORBOARD_DIR_BERT = 'tensorboard/BERT'
BERT_SAVED_MODEL_PATH = 'model/bert/'