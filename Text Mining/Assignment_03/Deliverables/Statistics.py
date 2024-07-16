import re
from wordcloud import WordCloud
from Hyperparameters import *
from DataPreprocess import TweetProcess
from scipy import stats, integrate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tp = TweetProcess()
text_len_dist, X_texts, Y_texts = tp.processTweets(statistics=True)

def tweet_sentiment_category_dist(file_path=DATASET_PATH, delimiter='\t'):
    pos, neg, neu = 0, 0, 0
    try:
            file = open(file_path, encoding = 'utf8')
            lines = file.readlines()
            for _, line in enumerate(lines):
                parse_word = line.strip().split(delimiter)
                if parse_word[1] == 'positive':
                    pos += 1
                elif parse_word[1] == 'negative':
                    neg += 1
                elif parse_word[1] == 'neutral':
                    neu += 1
    finally:
            if file:
                file.close()
    
    print('Positive count: ', pos)
    print('Negative count: ', neg)
    print('Neutral count: ', neu)

    plt.bar(['Positive'], [pos])
    plt.bar(['Negative'], [neg])
    plt.bar(['Neutral'], [neu])
    # plt.legend()
    plt.xlabel('Category of sentiment')
    plt.ylabel('Number of sentiment')
    plt.title('Distribution of sentiment category of tweet')
    plt.show()

def tweet_text_length_dist():

    # print(text_len_dist)
    # text_len_dist = list(text_len_dist)
    len_max = np.max(text_len_dist)
    len_min = np.min(text_len_dist)
    len_mean = np.mean(text_len_dist)
    len_median = np.median(text_len_dist)
    len_std = np.std(text_len_dist)
    len_count = np.sum(text_len_dist)
    print('Words Count: ', len_count)
    print('Max length of tweet: ', len_max)
    print('Min length of tweet: ', len_min)
    print('Mean length of tweet: ', len_mean)
    print('Median length of tweet: ', len_median)
    print('Standard deviation for the length of tweet: ', len_std)
    sns.kdeplot(text_len_dist, shade=True)
    plt.axvline(len_mean, label='mean',linestyle='-.', color='r')
    plt.axvline(len_median, label='median',linestyle='-.', color='g')
    plt.legend()
    plt.xlabel('Length of tweet range')
    plt.ylabel('Volume of tweet (density/percent)')
    plt.title('Distribution of length of tweet')
    plt.show()


def wordcount_gen(category):
                          
    wc = WordCloud(background_color='white', 
                   max_words=50, width=800, height=600)

    tweets = ' '.join([' '.join(item[0]) for item in zip(X_texts, Y_texts) if item[1] == category])
    # plt.figure(figsize=(10,10))
    plt.imshow(wc.generate(tweets))
    plt.title('<Wordcloud for {} sentiment>'.format(category), fontsize=20, color='grey')
    plt.axis('off')
    plt.show()

# tweet_sentiment_category_dist() # 1
# tweet_text_length_dist() # 2
# wordcount_gen('positive') # 3
