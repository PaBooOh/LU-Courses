from pickle import TRUE


SEED = 123 # for all functions that require random seed

# Dataset
DATASET_PATH = 'twitter.tsv'
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



# LSTM training
TEXT_LEN = 20 # Keras embedding
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS= 10
TENSORBOARD_DIR = 'tensorboard/'
LSTM_SAVED_MODEL_PATH = 'model/lstm/'