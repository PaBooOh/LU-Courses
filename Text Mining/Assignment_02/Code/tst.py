import nltk
import sklearn
import sklearn_crfsuite
import numpy as np
import pandas
from word2features import sent2labels, sent2features, sent2tokens, sent2features_custom
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
# from sklearn.cross_validation import cross_val_score
# from sklearn.grid_search import RandomizedSearchCV

def to_3arr(initial_conll_path, save=False, save_conll_path=None): 
    sentences = []
    sent_tmp = []   
    bio_tmp = []
    _3arr = []
    try:
        file = open(initial_conll_path, encoding = 'utf8')
        lines = file.readlines()
        for index, line in enumerate(lines):
            parse_word = line.strip().replace(u'\ufeff', '').split('\t')
            if len(parse_word) == 2:
                word = parse_word[0]
                word_bio = parse_word[1]
                sent_tmp.append(word)
                bio_tmp.append(word_bio)
                if index == len(lines) - 1:
                    sent_pos_tag = nltk.pos_tag(sent_tmp)
                    for index, item in enumerate(sent_pos_tag):
                        tmp_tup = item + (bio_tmp[index], )
                        _3arr.append(tmp_tup)
                    if save:
                        _3arr.append((np.nan,)) # a sentence end with NaN
                        sentences.extend(_3arr)
                    else:
                        sentences.append(_3arr)
            else:
                sent_pos_tag = nltk.pos_tag(sent_tmp)
                for index, item in enumerate(sent_pos_tag):
                    tmp_tup = item + (bio_tmp[index], )
                    _3arr.append(tmp_tup)
                if save:
                        _3arr.append((np.nan,)) # a sentence end with NaN
                        sentences.extend(_3arr)
                else:
                    sentences.append(_3arr)
                sent_tmp = []
                bio_tmp = []
                _3arr = []
    finally:
        if file:
            file.close()

    if save:
        df = pandas.DataFrame(data=sentences)
        df.to_csv(save_conll_path, index=None, header=None, sep='\t')
        return
    return sentences

train_sents = to_3arr('emerging.test.annotated')

# print(sent2features(train_sents[0])[-1])
# print(sent2features_custom(train_sents[0])[0])