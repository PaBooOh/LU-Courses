from tensorflow.python.ops.variables import trainable_variables
from DataPreprocess_O import TweetProcess
from Hyperparameters_O import *
from gensim.models import Word2Vec
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Activation, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.metrics import Precision, Recall, Accuracy
import numpy as np
import datetime

class Word2Vectors():
    def __init__(self) -> None:
        tp = TweetProcess()
        X, Y, vocabs, invocabs, X_numeric_text, tokenizer = tp.processTweets()

        self.X = X
        self.X_numeric_text = X_numeric_text
        self.Y = Y
        self.tokenizer = tokenizer
    
    def wordEmbeddings(self, saved_model_folder=W2V_MODEL_FOLDER, dim=WORD_DIM, window_size=WINDOW_SIZE, min_count=MIN_COUNT, sg=SG, epochs=W2V_EPOCHS):

        saved_model_file = saved_model_folder + 'w2v_dim' + str(dim) + '.ml'
        w2v_model = Word2Vec(self.X, min_count = min_count, vector_size = dim, window = window_size, sg = sg, epochs = epochs, seed=SEED)
        w2v_model.save(saved_model_file) # save word2vec model
    
    def loadW2VModel(self, loaded_model_file=W2V_SKIPGRAM_MODEL_FILE):
        self.w2v_model = Word2Vec.load(loaded_model_file)
    
    def getMostSim(self, word):
        self.loadW2VModel()
        return self.w2v_model.wv.most_similar(word)
    
    def getWordsIdx(self):
        self.loadW2VModel()
        print(self.w2v_model.wv.index_to_key)

    def getWordVecs(self, words):
        self.loadW2VModel()
        return self.w2v_model.wv[words]
    
    def getEmbeddingMatrix(self):
        tokenizer = self.tokenizer
        self.loadW2VModel()
        vocab_size = len(self.w2v_model.wv.key_to_index) # exclude <oov>
        embedding_matrix = np.zeros((vocab_size+1, WORD_DIM)) # +1 for <oov>

        for idx, word in tokenizer.index_word.items(): # idx of dict starts from 1
            if self.w2v_model.wv.__contains__(word):
                # print(idx, word)
                embedding_matrix[idx-1] = self.w2v_model.wv[word]
        
            # try:
            #     embedding_matrix[idx] = self.w2v_model.wv[word]
            # except:
            #     continue
        print('The shape of the embedding matrix: ', embedding_matrix.shape)
        self.embedding_matrix = embedding_matrix
        return embedding_matrix

class LSTMLayer():
    def __init__(self) -> None:
        w2v = Word2Vectors()
        self.embedding_mat = w2v.getEmbeddingMatrix()
        tp = TweetProcess()
        _, padding_X_train, padding_X_test, _, Y_train, Y_test = tp.kerasPadding()
        self.X_train = padding_X_train
        self.Y_train = Y_train
        self.X_test = padding_X_test
        self.Y_test = Y_test

    def training(self):
        lstm_model = Sequential()
        lstm_model.add(Embedding(
                    input_dim=self.embedding_mat.shape[0],
                    output_dim=WORD_DIM,
                    weights=[self.embedding_mat],
                    input_length=TEXT_LEN))
        lstm_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        lstm_model.add(MaxPooling1D(pool_size=2))
        lstm_model.add(Bidirectional(LSTM(32)))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(Dense(3, activation='softmax'))
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', 
               metrics=['acc', Precision(), Recall()])
        
        log_dir = TENSORBOARD_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        lstm_model.fit(
                    self.X_train, 
                    self.Y_train,
                    validation_data=(self.X_test, self.Y_test),
                    batch_size=LSTM_BATCH_SIZE, 
                    epochs=LSTM_EPOCHS,
                    callbacks = [tensorboard_callback],
                    verbose=1)
        
        lstm_model.save(LSTM_SAVED_MODEL_PATH + 'lstm_' + 'epoch' + str(LSTM_EPOCHS) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


lstm = LSTMLayer()
lstm.training()
# w2v = Word2Vectors()
# w2v.wordEmbeddings()

# print(w2v.getEmbeddingMatrix()[-1])
# print(w2v.getEmbeddingMatrix())
# train and get word embeddings

# w2v.getEmbeddingMatrix()
# w2v.getWordsIdx()
# print(w2v.getMostSim('bad'))