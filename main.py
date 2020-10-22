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
    - Prepcesses / clean data
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

# %% Import libraries
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

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

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

# %% Cleaning data
# Adapted from https://medium.com/towards-artificial-intelligence/blacklivesmatter-twitter-vader-sentiment-analysis-using-python-8b6e6fc2cd6a

def clean_tweets(tweets):
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    print("\tRemoving Retweet handles...")
    # Remove twitter Retweet handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:")

    print("\tRemoving username handles...")
    # Remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")

    print("\tRemoving URLs...")
    # Remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")

    print("\tRemoving double whitespaces...")
    # Remove multiple white spaces
    tweets = np.vectorize(remove_pattern)(tweets, "[\s][\s]+")

    print("\tRemoving punctuation...")
    # Remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")

    # Probably not the most elegant solution, but it works...
    tweets_ = []
    for row in tweets:
        row = row.lower()  # Make text lowercase
        tweets_.append(row)
    return tweets_

def loadData(filename, csv_path, columns, samples=800000-1, mapping={}, drop_columns=[]):
    try:
        with open(filename, 'rb') as file:
            df = pickle.load(file)
            tprint("Loaded dataframe.")
    except FileNotFoundError:
        tprint(f"The file \"{filename}\" was not found.")
        samples = min(samples, 800000 - 1)

        tprint(f"Loading data ({samples * 2} datapoints)... ", end='', flush=True)
        df = pd.read_csv(csv_path, names=columns, encoding='ISO-8859-1')
        df = df.drop(drop_columns, axis=1)

        # Better way of sampling, this way have guaranteed balanced data.
        df = pd.concat([df.query("Sentiment==0").sample(samples), df.query("Sentiment==4").sample(samples)])
        df['Sentiment'] = df['Sentiment'].map(mapping)  # map 4 to 1
        print(time_(), "Done.")

        print(time_(), "Cleaning tweets...")
        df['Text'] = clean_tweets(df['Text'])
        with open('clean_tweets.txt', 'wb') as file:
            pickle.dump(df, file)
        tprint(f"Dataframe stored in \"{filename}\".")
    return df.sample(samples)


def tprint(str, end='\n', flush=True):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] -", str, end=end, flush=flush)

class Scores:
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

    # # %% Obtain FT F1, acc and roc values
    # def predict(row):
    #     return model.predict(row['Text'])

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

def LR(vectoriser=None, stem=False, stopwords=True, n=1, parameters=None):
    tprint("Logistic Regression is starting...")

    def tokenize(text):
        '''Used in initialising the TweetTokenizer.'''
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    def stem(doc):
        '''Stems words to reduce token variations in sentences.'''
        return (SnowballStemmer.stem(w) for w in analyzer(doc))

    if vectoriser is None:
        vectoriser = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=True,
            ngram_range=(n, n),
            stop_words=(en_stopwords if stopwords else None))

    # Parameter optimization
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Linear kernel since it is a binary problem (pos, neg)
    pipeline_LR = make_pipeline(vectoriser, LogisticRegression(max_iter=1000))

    # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
    if parameters is None:
        param_grid_ = {
            'logisticregression__penalty': ['l1', 'l2'],
            'logisticregression__C': np.logspace(-4, 4, 20),
            'logisticregression__solver': ['liblinear']}  # liblinear; saga often the best choice but takes way more time
    else:
        # TODO: Fix model parameters to reduce runtime.
        pass

    # Create the grid search object to find optimal hyperparameters. n_jobs may be varied to use multiple CPU cores.
    # Note that this may not necessarily be faster, depending on the classification task!
    grid_LR = GridSearchCV(pipeline_LR,
                           param_grid=param_grid_,
                           cv=kfolds,
                           scoring="roc_auc",
                           verbose=1,
                           n_jobs=4)

    grid_LR.fit(X_train_, y_train_)
    grid_LR.score(X_test_, y_test_)
    print('Best LR paramater:' + str(grid_LR.best_params_))
    print('Best score: ' + str(grid_LR.best_score_))

    model = grid_LR.best_estimator_
    pred = model.predict(X_test_)

    lr_scores = Scores(pred, y_test_, "LR")
    tprint("Logistic regression is done.")
    pprint(lr_scores.get_dict())
    return lr_scores


## Naive Bayes: MultinomialNB with unigrams and TF-IDF

def NB(vectorizer=None, stem=False, stopwords=True, n=1, clf=None):
    tprint("Naive Bayes is starting...")
    def tokenize(text):
        '''Used in initialising the TweetTokenizer.'''
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    if clf is None:
        clf = BernoulliNB()

    tk = TweetTokenizer()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=True,
            ngram_range=(n, n),
            stop_words=(en_stopwords if stopwords else None)
        )

    text_counts = vectorizer.fit_transform(df['Text'])

    X_train, X_test, y_train, y_test = train_test_split(text_counts, df['Sentiment'], test_size=0.2,
                                                        random_state=42)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    nb_scores = Scores(y_test, prediction, "NB")

    tprint("Naive Bayes is done.")
    pprint(nb_scores.get_dict())
    return nb_scores


## Bar chart for f1 and acc scores for all classifiers
def results(score_dict):

    def round_vals(input):
        output = []
        for num in input:
            num = round(num * 100, 2)
            output.append(num)
        return output

    def plot():

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
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
        ax.set_title('Accuracy and F1 scores per classifier')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right')
        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

    def plot_roc():
        fig, ax = plt.subplots()

        # Plot ROC curves for all classifers in one graph
        plt.plot(score_dict["LR"].get_dict()["FPR"], score_dict["LR"].get_dict()["TPR"], color="red")
        plt.plot(score_dict["VADER"].get_dict()["FPR"], score_dict["VADER"].get_dict()["TPR"], color="blue")
        plt.plot(score_dict["FT"].get_dict()["FPR"], score_dict["FT"].get_dict()["TPR"], color="pink")
        plt.plot(score_dict["NB"].get_dict()["FPR"], score_dict["NB"].get_dict()["TPR"], color="cyan")
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(list(score_dict.keys()))
        plt.show()

    accs = [clf.get_dict()["ACC"] for clf in score_dict.values()]
    accs = round_vals(accs)
    F1 = [clf.get_dict()["F1"] for clf in score_dict.values()]
    F1 = round_vals(F1)

    plot()
    plot_roc()

##------------------------------##

filename = "clean_tweets.txt"
target_file = "classifier_scores.txt"
csv_path = r"16mtweets.csv"
columns = ['Sentiment', 'ID', 'Date', 'Flag', 'User', 'Text']
samples = 40000  # Samples per class (in this case half of the length of the dataset)

mapping = {0: 0, 4: 1}  # Maps 0 to 0, and 4 to 1. The number 1, 2, and 3 don't seem to occur in the dataset.
drop_columns = ['ID', 'Flag', 'User']  # Unused columns.

df = loadData(filename, csv_path, columns, samples, mapping, drop_columns)

# Smaller sample for SVM / LR due to processing time.
sample_size = 10000
df2 = pd.concat([df.query("Sentiment==0").sample(sample_size), df.query("Sentiment==1").sample(sample_size)])

## Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)
X_train_, X_test_, y_train_, y_test_ = train_test_split(df2['Text'], df2['Sentiment'], test_size=0.2, random_state=42)

stemmer = SnowballStemmer('english')
tk = TweetTokenizer()

# Set your arguments here inside the respective classifier functions

scores = {
    "VADER": VADER(),
    "FT": FT(),
    "LR": LR(),
    "NB": NB()
}

results(scores)

# with open(target_file, 'w') as file:
#     file.write(json.dumps(results))
#     tprint(f"Wrote results to \"{target_file}\"")
