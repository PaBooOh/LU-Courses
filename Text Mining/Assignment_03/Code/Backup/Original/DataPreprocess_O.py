from pickle import FALSE
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from Hyperparameters_O import *
import re
import numpy as np

# negation_handling = {
#     "aren't" : 'are not',
#     "isn't" : 'is not',
#     "wasn't" : 'was not',
#     "weren't" : 'were not',
# }

class TweetProcess():

    def __init__(self) -> None:
        pass
    
    def processTweets(self, file_path = DATASET_PATH, delimiter='\t'):
        # dataset = pd.read_csv(file_path, delimiter=delimiter, header=None)
        # rm_http = lambda x: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(x))
        # rm_at = lambda x: re.sub('[!:*.,-]|[@]','',str(x))
        X = []
        Y = []
        vocabs, idx = dict(), 1
        max_len, sum_len = 0, 0
        vocabs['<oov>'] = 0
        stemmer = PorterStemmer()
        try:
            file = open(file_path, encoding = 'utf8')
            lines = file.readlines()
            for _, line in enumerate(lines):
                parse_word = line.strip().split(delimiter)
                if parse_word[1] == 'positive':
                    parse_word[1] = 2
                elif parse_word[1] == 'negative':
                    parse_word[1] = 0
                elif parse_word[1] == 'neutral':
                    parse_word[1] = 1
                # Regex
                parse_word[2] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', parse_word[2])
                parse_word[2] = re.sub('@([a-zA-Z_0-9])+', '' , parse_word[2])
                parse_word[2] = re.sub('[^#a-zA-z0-9\' ]', '', parse_word[2])
                parse_word[2] = re.sub('[\[\]]', '', parse_word[2])
                parse_word[2] = re.sub('[0-9][a-zA-Z]+', '', parse_word[2]) # remove digits
                parse_word[2] = re.sub('[a-zA-Z]+[0-9]+', '', parse_word[2]) # remove digits
                parse_word[2] = re.sub('[0-9]+', '', parse_word[2]) # remove digits
                parse_word[2] = re.sub("['][a-zA-z0-9]+", '', parse_word[2]) 
                parse_word[2] = re.sub('amp', '', parse_word[2])
                parse_word[2] = parse_word[2].lower()
                parse_word[2] = parse_word[2].split()
                parse_word[2] = [word for word in parse_word[2] if len(word) >= 3]
                # parse_word[2] = [stemmer.stem(word) for word in parse_word[2] if word not in stopwords.words('english')]
                for token in parse_word[2]:
                    if token not in vocabs:
                        vocabs[token] = idx
                        idx += 1
                X.append(parse_word[2]) # Dataset: sentences
                Y.append(parse_word[1]) # Labels
                # calculate two kinds of length of Texts for 
                if len(parse_word[2]) > max_len:
                    max_len = len(parse_word[2])
                sum_len += len(parse_word[2])
        finally:
            if file:
                file.close()
                
        invocabs = {idx: token for token, idx in vocabs.items()}
        Y = to_categorical(Y, num_classes=3, dtype='int') # one-hot
        print('The maximum length of texts is: ', max_len)
        print('The average length of texts is: ', sum_len/len(Y))

        tokenizer = Tokenizer(oov_token="<oov>")
        tokenizer.fit_on_texts(X)
        X_numeric_text = tokenizer.texts_to_sequences(X)

        self.X_numeric_texts = X_numeric_text
        self.tokenizer = tokenizer
        self.X_texts = X
        self.Y = Y
        self.vocabs = vocabs
        self.invocabs = invocabs
        return X, Y, vocabs, invocabs, X_numeric_text, tokenizer
    
    def kerasPadding(self):
        tp = TweetProcess()
        _, Y, _, _, X_numeric_text, _ = tp.processTweets()
        X_train, X_test, Y_train, Y_test = TweetProcess.splitTweets(X_numeric_text, Y)
        
        padding_X = pad_sequences(X_numeric_text, maxlen=TEXT_LEN, padding='post')
        padding_X_train = pad_sequences(X_train, maxlen=TEXT_LEN, padding='post')
        padding_X_test = pad_sequences(X_test, maxlen=TEXT_LEN, padding='post')
        return padding_X, padding_X_train, padding_X_test, Y, Y_train, Y_test
    

    @staticmethod
    def splitTweets(X, Y, test_size=TEXT_SIZE, random_state=SEED):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        return X_train, X_test, Y_train, Y_test
    
# tp = TweetProcess()
# X, Y, vocabs, invocabs, X_numeric_text, tokenizer = tp.processTweets()
# padding_X, padding_X_train, padding_X_test, Y, Y_train, Y_test = tp.kerasPadding()
# print(Y_train.shape)
# print(padding_X_train.shape)
# small_word_index = copy.deepcopy(word_index)
# X, Y, vocabs, invocabs  = TweetProcess.processTweets()
