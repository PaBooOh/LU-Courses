from DataPreprocess import TweetProcess
from Hyperparameters import *
from gensim.models import Word2Vec
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import Input
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, TFBertModel
import os
import matplotlib.pyplot as plt
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
        # import time

        saved_model_file = saved_model_folder + 'w2v_dim' + str(dim) + '_all.ml'
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
        # print('The shape of the embedding matrix: ', embedding_matrix.shape)
        self.embedding_matrix = embedding_matrix
        return embedding_matrix

class NaiveBayes():
    def __init__(self, representation_flag):
        self.representation_flag = representation_flag
        tp = TweetProcess()
        self.tp = tp
        # if self.representation_flag == 'bow':
        _, X, Y = tp.processTweets(statistics=True)
        self.X = X
        self.Y = Y

    def training_bow(self):
        assert(self.representation_flag == 'bow')
        cv = CountVectorizer(max_features=MAX_VOCABS)
        X = [' '.join(text) for text in self.X]
        X = cv.fit_transform(X).toarray()
        X_train, X_test, Y_train, Y_test = self.tp.splitTweets(X, self.Y)
        NB_clf = MultinomialNB()
        NB_clf.fit(X_train, Y_train)
        pred = NB_clf.predict(X_test)
        target_names = ['negative', 'neutral', 'positive']
        bow_acc = accuracy_score(Y_test, pred)
        bow_score = classification_report(Y_test, pred, target_names=target_names)
        plot_confusion_matrix(NB_clf, X_test, Y_test)
        print('Accuracy: ', bow_acc)
        print(bow_score)
        plt.show()
    
    def training_tfidf(self):
        assert(self.representation_flag == 'tfidf')
        tfidf = TfidfVectorizer(max_features=MAX_VOCABS, ngram_range=N_GRAM_IDF, binary=BINARY_IDF, smooth_idf=SMOOTH_IDF)
        X = [' '.join(text) for text in self.X]
        X = tfidf.fit_transform(X).toarray()
        X_train, X_test, Y_train, Y_test = self.tp.splitTweets(X, self.Y)
        NB_clf = MultinomialNB()
        NB_clf.fit(X_train, Y_train)
        pred = NB_clf.predict(X_test)
        target_names = ['negative', 'neutral', 'positive']
        tfidf_acc = accuracy_score(Y_test, pred)
        tfidf_score = classification_report(Y_test, pred, target_names=target_names)
        plot_confusion_matrix(NB_clf, X_test, Y_test)
        print('Accuracy: ', tfidf_acc)
        print(tfidf_score)
        plt.show()

class LSTMLayer():

    def training_w2v(self):
        w2v = Word2Vectors()
        self.embedding_mat = w2v.getEmbeddingMatrix()
        optimizer = Adam(learning_rate=LSTM_LR)
        tp = TweetProcess()
        _, padding_X_train, padding_X_test, _, Y_train, Y_test = tp.kerasPadding()
        self.X_train = padding_X_train
        self.Y_train = Y_train
        self.X_test = padding_X_test
        self.Y_test = Y_test
        # bi-lstm
        lstm_model = Sequential()
        lstm_model.add(Embedding(
                    input_dim=self.embedding_mat.shape[0],
                    output_dim=WORD_DIM,
                    weights=[self.embedding_mat],
                    input_length=TEXT_LEN))
        lstm_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        lstm_model.add(MaxPooling1D(pool_size=2))
        lstm_model.add(Bidirectional(LSTM(32)))
        lstm_model.add(Dropout(0.5))
        lstm_model.add(Dense(3, activation='softmax'))
        lstm_model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer, 
            metrics=['acc', Precision(), Recall()])
        
        log_dir = TENSORBOARD_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        lstm_model.fit(
                    self.X_train, 
                    self.Y_train,
                    validation_data=(self.X_test, self.Y_test),
                    batch_size=LSTM_BATCH_SIZE, 
                    epochs=LSTM_EPOCHS,
                    callbacks=[tensorboard_callback],
                    verbose=1)
        
        lstm_model.save(LSTM_SAVED_MODEL_PATH + 'lstm_' + 'epoch' + str(LSTM_EPOCHS) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def load_model(self):
        from tensorflow.keras.models import load_model
        return load_model(LSTM_LOADED_MODEL_PATH)
    

class BertLayer():
    def __init__(self) -> None:
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def tokenize(self, X_texts):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_ids, attention_masks = [], []
        for text in X_texts:
            encoded_sent = self.bert_tokenizer.encode_plus(
            text=' '.join(text),
            add_special_tokens=True,
            max_length=MAX_LEN_BERT,
            pad_to_max_length=True,
            # padding='max_length',
            return_attention_mask=True
        )
            input_ids.append(encoded_sent['input_ids'])
            attention_masks.append(encoded_sent['attention_mask'])
        return np.array(input_ids), np.array(attention_masks)
    
    def encode_data(self):
        tp = TweetProcess()
        X, Y, _, _, _, _ = tp.processTweets()
        X_train, X_test, Y_train, Y_test = tp.splitTweets(X, Y)
        train_input_ids, train_attention_masks = self.tokenize(X_train)
        test_input_ids, test_attention_masks = self.tokenize(X_test)
        return train_input_ids, train_attention_masks, test_input_ids, test_attention_masks, Y_train, Y_test
    
    def fine_tune_bert(self):
        optimizer = Adam(learning_rate=LR_BERT, decay=1e-7)

        input_ids = Input(shape=(MAX_LEN_BERT,), dtype='int32')
        attention_masks = Input(shape=(MAX_LEN_BERT,), dtype='int32')
        embeddings = self.bert_model([input_ids, attention_masks])[1]
        output = Dense(3, activation="softmax")(embeddings)
        model = Model(inputs = [input_ids, attention_masks], outputs = output)
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer, 
            metrics=['acc', Precision(), Recall()])
        
        log_dir = TENSORBOARD_DIR_BERT + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        X_train_input_ids, X_train_attention_masks, X_test_input_ids, X_test_attention_masks, Y_train, Y_test = self.encode_data()
        model.fit(
            [X_train_input_ids, X_train_attention_masks],
            Y_train,
            validation_data=([X_test_input_ids, X_test_attention_masks], Y_test),
            batch_size=BATCH_SIZE_BERT,
            epochs=EPOCHS_BERT,
            callbacks=[tensorboard_callback]
        )

        # model.save(BERT_SAVED_MODEL_PATH + 'model_' + 'epoch' + str(LSTM_EPOCHS)+ '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


'''
Training
'''
# ---> word feature learning
# 1) word2vec
# w2v = Word2Vectors()
# w2v.wordEmbeddings()

# ---> clf training
# 1) Bi-lstm on word2vec (need to use Word2vec to generate words embeddings first)
# lstm = LSTMLayer()
# lstm.training_w2v()

# 2) NB (baseline) on boW or tfidf 
# nb_clf = NaiveBayes('bow')
# nb_clf.training_bow()
# nb_clf.training_tfidf()

# 3) BERT
bert = BertLayer()
bert.fine_tune_bert()
