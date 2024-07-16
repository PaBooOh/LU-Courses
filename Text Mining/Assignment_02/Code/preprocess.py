import nltk
import sklearn
import sklearn_crfsuite
import numpy as np
import pandas
from word2features import sent2labels, sent2features, sent2tokens
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

train_sents = to_3arr('wnut17train.conll')
test_sents = to_3arr('emerging.test.annotated')

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)



labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(X_test)

print(metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

# train
# to_3arr('wnut17train.conll', 'wnut17train_new.conll')
# test
# to_3arr('emerging.test.annotated', 'emerging.test_new.annotated')
# dev
# to_3arr('emerging.dev.conll', 'emerging.dev_new.conll')

#def to_3arr(initial_conll_path, save_conll_path):
#     import csv
#     sets = pandas.read_table(initial_conll_path, delimiter='\t', header=None, engine="python", encoding='utf-16')
#     sentences = []
#     sent_tmp = []   
#     bio_tmp = []
#     _3arr = []
#     for index, row in sets.iterrows():
#         if row[0] != row[0]: # denotes a sentence extracted successfully
#             sent_pos_tag = nltk.pos_tag(sent_tmp)
#             for index, item in enumerate(sent_pos_tag):
#                 tmp_tup = item + (bio_tmp[index], )
#                 _3arr.append(tmp_tup)
#             _3arr.append((np.nan, np.nan, np.nan)) # a sentence end with NaN
#             sentences.extend(_3arr)
#             sent_tmp = []
#             bio_tmp = []
#             _3arr = []
#             continue
#         sent_tmp.append(row[0])
#         bio_tmp.append(row[1])
#     df = pandas.DataFrame(data=sentences)
#     df.to_csv(save_conll_path, index=None)