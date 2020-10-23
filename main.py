# -*- coding: utf-8 -*-
"""
Sentiment analysis of geotagged tweets with location USA over time.

The training dataset (sentiment values of tweets) can be found here:
https://www.kaggle.com/kazanova/sentiment140

The initial data set for tweeets of the usa can be found here: https://ieee-dataport.org/open-access/coronavirus-covid-19-geo-tagged-tweets-dataset
The tweet IDs in this data set need then to be hydrated (e.g. using twarc) and filtered for the location "USA".

The cells contain the following steps:
    - Load libraries
    - Loads data
    - Preprocesses / clean data
    - Split data into test and train
    - Classify
        - Vader
        - Fast Text
        - Logistic Regression
        - Naive Bayes
    - Plot Bar chart for f1 and acc scores
    - Plot ROC curves

TODO:
    - plot sentiment over time for the tweets from usa using the best classifier.
"""

import re
import pandas as pd
import numpy as np
import glob
import string
import nltk
import sklearn
import math
import matplotlib.pyplot as plt
import fasttext
import seaborn as sns
from pprint import pprint
import json
from datetime import datetime
import pickle

from nltk.corpus import stopwords
nltk.download('stopwords')
en_stopwords = set(stopwords.words("english"))

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
from textblob import TextBlob

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB, BernoulliNB

def loadData(filename, csv_path, columns, samples=800000-1, mapping={}, drop_columns=[], stopwordFiltering=False):
    '''
    Loads in previously cleaned data set ``filename``, or cleans new data set using the rest of the parameters and stores
    it in ``filename`` for later reuse.

    :param filename: The pickled file with previously cleaned data.
    :param csv_path: The .csv file with the original data. Used if ``filename`` cannot be found.
    :param columns: Headers of the columns that should be kept.
    :param samples: Amount of samples per class. Should be max amount of data samples / 2.
    :param mapping: Map labels to other values in a dictionary.
    :param drop_columns: Columns that are not of interest can be dropped.
    :return: pd.DataFrame of length=2*samples.
    '''
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    def get_pos(word):
        '''
        Get part-of-speech tag from WordNet. Limited version, more tags are available.
        :param word: input string
        :return: Tuple of the form (word, WordNet POS-tag)
        '''
        # From https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizerwithappropriatepostag
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)  # Return NOUN as default

    def lemmatise_(tweet):
        '''
        Lemmatise an entire sentence based on automated POS tagging
        :param tweet: input string
        :return: lemmatised string
        '''
        sentences = tweet.split('.!?;')  # Split into seperate sentences
        new_tweet = []
        for sentence in sentences:
            word_list = word_tokenize(sentence)
            lemmatized = ' '.join([lem.lemmatize(w, get_pos(w)) for w in word_list])
            new_tweet.append(lemmatized)
        tweet = ' '.join(new_tweet)
        return tweet

    def correct_spelling(tweet):
        '''
        Automated spelling correction
        :param tweet: input string
        :return: spelling corrected string
        '''
        return TextBlob(str(tweet).lower()).correct().raw

    def lowercase(tweet):
        return tweet.lower()

    def stopwordFilter(tweet):
        word_list = word_tokenize(tweet)
        filtered = [word for word in word_list if not word in en_stopwords]
        return ' '.join(filtered)

    def clean_tweet(tweets):
        '''
        Removes RT handles, username handles, URLs, double whitespaces and finally interpunction
        :param tweet: input tweets
        :return: List of input strings fully cleaned of unwanted artefacts.
        '''
        # Adapted from https://medium.com/towards-artificial-intelligence/blacklivesmatter-twitter-vader-sentiment-analysis-using-python-8b6e6fc2cd6a

        tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:")
        tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
        tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")
        tweets = np.vectorize(remove_pattern)(tweets, "[\s][\s]+")
        # tweets = np.vectorize(correct_spelling)(tweets)
        # tweets = np.vectorize(lemmatise_)(tweets)
        tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
        tweets = np.vectorize(lowercase)(tweets)
        if (stopwordFiltering):
            tweets = np.vectorize(stopwordFilter)(tweets)

        # Probably not the most elegant solution, but it works...
        # tweets_ = []
        # for tweet in tweets:
        #     tweet = row.lower()  # Make text lowercase
        #     # corrected = TextBlob(row).correct()  # Correct spelling using TextBlob.
        #     tweets_.append(tweet)

        return tweets

    lem = WordNetLemmatizer()

    if stopwordFiltering:
        filename = filename + "_no_stopwords"
    try:
        with open(filename, 'rb') as file:

            df = pickle.load(file)
            tprint("Loaded dataframe.")
    except Exception:
        tprint(f"The file \"{filename}\" was not found.")
        samples = min(samples, 800000 - 1)

        tprint(f"Loading data ({samples * 2} datapoints)... ", end='', flush=True)
        df = pd.read_csv(csv_path, names=columns, encoding='ISO-8859-1')
        df = df.drop(drop_columns, axis=1)

        # Better way of sampling, this way have guaranteed balanced data.
        df = pd.concat([df.query("Sentiment==0").sample(samples), df.query("Sentiment==4").sample(samples)])
        df['Sentiment'] = df['Sentiment'].map(mapping)  # map 4 to 1
        tprint("Done.")

        tprint("Cleaning tweets...")
        # for i in range(len(df['Text'])):
        #     df['Text'].iloc[i] = clean_tweet(df['Text'].iloc[i])
        df['Text'] = clean_tweet(df['Text'])

        with open(filename, 'wb') as file:
            pickle.dump(df, file)
        tprint(f"Dataframe stored in \"{filename}\".")
    return df.sample(samples)


def tprint(str, end='\n', flush=True):
    '''
    Print function with [HH:MM:SS] timestamp, used for timing.
    :param str: input string
    :param end: End-of-line char. Should be '' if another string is to be printed on the same line.
    :param flush: Flush the output buffer
    :return: None
    '''
    print(f"[{datetime.now().strftime('%H:%M:%S')}] -", str, end=end, flush=flush)

class Scores:
    '''
    Wrapper object for all score metrics relevant to us.
    '''
    def __init__(self, prediction, ground_truth, clf):
        self.clf = clf
        self.pred = prediction
        self.ground_truth = ground_truth

        # Calculate false positive rate and true positive rate.
        self.fpr, self.tpr, self.thresholds = roc_curve(ground_truth, prediction)
        self.roc_auc = roc_auc_score(ground_truth, prediction)
        self.f1 = f1_score(ground_truth, prediction)
        self.acc = accuracy_score(ground_truth, prediction)
        self.precision = precision_score(ground_truth, prediction)
        self.recall = recall_score(ground_truth, prediction)

    def get_dict(self):
        d = {}
        d["FPR"] = self.fpr
        d["TPR"] = self.tpr
        d["AUC"] = self.roc_auc
        d["F1"] = self.f1
        d["ACC"] = self.acc
        d["PRC"] = self.precision
        d["REC"] = self.recall
        return d


## VADER
# Adapted from https://medium.com/towards-artificial-intelligence/blacklivesmatter-twitter-vader-sentiment-analysis-using-python-8b6e6fc2cd6a
def VADER():
    tprint("Starting VADER...")
    analyzer = SentimentIntensityAnalyzer()

    # We don't need to train, since this is a pre-trained model!
    # Use compound values, those are the normalized values from pos, neg and neutral. Range: [-1,1]
    vs_predictions = [analyzer.polarity_scores(row)["compound"] for row in X_test]
    vs_predictions = pd.cut(vs_predictions, bins=2, labels=[0, 1])  # Map cont. values from [-1,1] to either 0 or 1.
    vader_scores = Scores(y_test, vs_predictions, 'VADER')
    tprint("VADER is done.")
    pprint(vader_scores.get_dict())
    return vader_scores


## FastText
def FT(n=2):

    tprint("FastText is starting...")
    def strip_FT(input):
        # Removes label from FT data and converts to int.
        output = []
        for row in input:
            line = ''.join(str(x) for x in row)
            line = re.sub('__label__', '', line)
            line = int(line)
            output.append(line)
        return output

    # Append __label__0 prefix for fastText, so it recognizes it as a label and not a word
    y_train_ft = "__label__" + y_train.astype(str)  # append_labels(y_train)
    y_test_ft = "__label__" + y_test.astype(str)

    ft_train = pd.concat([X_train, y_train_ft], axis=1, sort=False)
    ft_test = pd.concat([X_test, y_test_ft], axis=1, sort=False)
    ft_train.to_csv('ft_train.txt', sep='\t', index=False, header=False)  # FastText needs a .txt file as input
    ft_test.to_csv('ft_test.txt', sep='\t', index=False, header=False)

    # These parameters are not necessarily optimal.
    hyper_params = {"lr": 0.01,
                    "epoch": 20,
                    "wordNgrams": n,
                    "dim": 20}

    # Train the model
    model = fasttext.train_supervised(input='ft_train.txt', verbose=False, **hyper_params)
    # optimization: https://notebook.community/fclesio/learning-space/Python/fasttext-autotune
    # model = fasttext.train_supervised(input='ft_train.txt', autotuneValidationFile='ft_test.txt')
    # print("Model trained with the hyperparameter \n {}".format(hyper_params))

    # Evaluate the model
    result = model.test('ft_train.txt')
    validation = model.test('ft_test.txt')

    ''' 
    # Display accuracy. I think it is actually F1
    text_line = "\n Hyper paramaters used:\n" + str(hyper_params) + ",\n training accuracy:" + str(result[1])  + ", \n test accuracy:" + str(validation[1]) + '\n' 
    print(text_line)


    def print_results(N, p, r):
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))

    print_results(*result)
    print_results(*validation)
    '''

    ft_pred = ft_test.apply(lambda row: model.predict(row['Text']), axis=1)

    ft_pred = pd.DataFrame(ft_pred)  # convert from series to df
    ft_pred = ft_pred[0].str.get(0)  # get first tuple

    ft_pred_stripped = strip_FT(ft_pred)
    ft_test_stripped = strip_FT(ft_test['Sentiment'])

    ft_scores = Scores(ft_test_stripped, ft_pred_stripped, 'FastText')
    tprint("FastText is done.")
    pprint(ft_scores.get_dict())
    return ft_scores


## Logistic Regression
# Adopted from https://www.kaggle.com/lbronchal/sentiment-analysis-with-svm
def LR(vectoriser=None, stopwords=False, n=1, hyperparams=None):
    '''
    Logistic regression classifier, trained on df data set.
    :param vectoriser: Defaults to TfidfVectoriser
    :param stopwords: Remove stopwords, defaults to False
    :param n: length of n-gram, defaults to True
    :param hyperparams: Dictionary of hyperparameters. Can be found using grid search if left blank.
    :return: Scores
    '''

    def tokenize(text):
        '''
        Used in initialising the TweetTokenizer.
        '''
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    if vectoriser is None:
        vectoriser = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=True,
            ngram_range=(n, n),
            stop_words=(en_stopwords if stopwords else None))

    tprint("Logistic Regression is starting...")

    text_counts = vectoriser.fit_transform(df2['Text'])
    X_train, X_test, y_train, y_test = train_test_split(text_counts, df['Sentiment'], test_size=0.2,
                                                        random_state=42)

    if hyperparams is None:
        # Parameter optimization
        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        # Linear kernel since it is a binary problem (pos, neg)
        pipeline_LR = make_pipeline(vectoriser, LogisticRegression(max_iter=1000))

        # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
        param_grid_ = {
            'logisticregression__penalty': ['l1', 'l2'],
            'logisticregression__C': np.logspace(-4, 4, 20),
            'logisticregression__solver': ['liblinear']}  # liblinear; saga often the best choice but takes way more time
        # Create the grid search object to find optimal hyperparameters. n_jobs may be varied to use multiple CPU cores.
        # Note that this may not necessarily be faster, depending on the classification task!
        grid_LR = GridSearchCV(pipeline_LR,
                               param_grid=param_grid_,
                               cv=kfolds,
                               scoring="roc_auc",
                               verbose=1,
                               n_jobs=4)
        grid_LR.fit(X_train, y_train)
        grid_LR.score(X_test, y_test)
        print('Best LR paramater:' + str(grid_LR.best_params_))
        print('Best score: ' + str(grid_LR.best_score_))
        model = grid_LR.best_estimator_
    else:
        model = LogisticRegression(**hyperparams)
        model.fit(X_train, y_train)

    pred = model.predict(X_test)

    lr_scores = Scores(pred, y_test, "LR")
    tprint("Logistic regression is done.")
    pprint(lr_scores.get_dict())
    return lr_scores


# Naive Bayes
def NB(vectoriser=None, stopwords=False, n=1, clf=None):
    '''
    Train a Naive Bayes classifier with df as input.
    :param vectoriser: Defaults to TfidfVectoriser
    :param stopwords: Remove stopwords, defaults to False
    :param n: length of n-gram, defaults to True
    :param clf: Type of NB classifier. Defaults to BernoulliNB.
    :return: Scores
    '''
    tprint("Naive Bayes is starting...")

    def tokenize(text):
        '''
        Used in initialising the TweetTokenizer.
        '''
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    if clf is None:
        clf = BernoulliNB()

    tk = TweetTokenizer()

    if vectoriser is None:
        vectoriser = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=True,
            ngram_range=(n, n),
            stop_words=(en_stopwords if stopwords else None)
        )

    text_counts = vectoriser.fit_transform(df2['Text'])
    X_train, X_test, y_train, y_test = train_test_split(text_counts, df['Sentiment'], test_size=0.2,
                                                        random_state=42)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    nb_scores = Scores(y_test, prediction, "NB")

    tprint("Naive Bayes is done.")
    pprint(nb_scores.get_dict())
    return nb_scores


def results(score_dict):
    '''
    Plot all accuracies and F1-scores of each classifier.
    :param score_dict: Dictionary in the form {classifier: Scores object}
    :return: None
    '''

    def round_vals(input):
        output = []
        for num in input:
            num = round(num * 100, 2)
            output.append(num)
        return output

    def plot():
        '''
        Plots bar charts of accuracy and F1 metrics for each classifier in ``score_dict``.
        :return: None
        '''

        def autolabel(rects):
            '''
            Attach a text label above each bar in *rects*, displaying its height.
            :param rects: Bar chart ``rect`` object.
            :return: None
            '''

            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        labels = score_dict.keys()
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, accs, width, label='Acc', color='#4a1d7a')
        rects2 = ax.bar(x + width / 2, F1, width, label='F1', color='#ac71ec')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(f'Accuracy and F1 scores (n={samples})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right')
        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

    def plot_roc():
        '''
        Plots the Receiver Operation Curve for each classifier in ``score_dict``.
        :return: None
        '''

        fig, ax = plt.subplots()

        colors = {"VADER": "blue", "FT": "pink", "NB":"cyan", "LR":"red"}
        # Plot ROC curves for all classifers in one graph
        for clf in score_dict.keys():
            plt.plot(score_dict[clf].get_dict()["FPR"], score_dict["LR"].get_dict()["TPR"], color=colors[clf])
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC (n={samples})')
        plt.legend(list(score_dict.keys()))
        plt.show()

    accs = [clf.get_dict()["ACC"] for clf in score_dict.values()]
    accs = round_vals(accs)
    F1 = [clf.get_dict()["F1"] for clf in score_dict.values()]
    F1 = round_vals(F1)

    plot()
    plot_roc()

##------------------------------##

filename = "clean_tweets"
target_file = "classifier_scores.txt"
csv_path = r"16mtweets.csv"
columns = ['Sentiment', 'ID', 'Date', 'Flag', 'User', 'Text']
mapping = {0: 0, 4: 1}  # Maps 0 to 0, and 4 to 1. The number 1, 2, and 3 don't seem to occur in the dataset.
drop_columns = ['ID', 'Flag', 'User']  # Unused columns.

samples = 800000  # Samples per class (in this case half of the length of the dataset)
samples2 = 800000  # Smaller sample for SVM / LR due to processing time.

# Found using grid search using 40.000 samples, now fixed to reduce processing time.
LR_hyper = {'penalty':'l2', 'C':0.23357214690901212, 'solver':'liblinear', 'max_iter':1000}
vectoriser = None  # Fixed to TfidfVectoriser
n=1  # Fixed to n=1
filterStopwords = True

## --------------------------------- ##

df = loadData(filename, csv_path, columns, samples, mapping, drop_columns, filterStopwords)
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

if samples == samples2:
    df2 = df
else:
    df2 = pd.concat([df.query("Sentiment==0").sample(samples), df.query("Sentiment==1").sample(samples)])
    X_train_, X_test_, y_train_, y_test_ = train_test_split(df2['Text'], df2['Sentiment'], test_size=0.2, random_state=42)

tk = TweetTokenizer()

scores = {
    'VADER': VADER(), # No tweaking
    'FT': FT(n=n),
    'LR': LR(vectoriser=None, n=n, hyperparams=LR_hyper, stopwords=False),
    'NB': NB(vectoriser=None, n=n, stopwords=False)
}

results(scores)

# with open(target_file, 'w') as file:
#     file.write(json.dumps(results))
#     tprint(f"Wrote results to \"{target_file}\"")
