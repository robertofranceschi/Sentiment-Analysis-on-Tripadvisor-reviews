# Project assignment - Winter Session, A.Y. 2019/2020
# Data Science Lab: process and method - Politecnico di Torino

# Student: Roberto Franceschi

# BEST ACCURACY FOUND
# Submission: 2020-01-22 23:32:48.334063
# 0.9746452184675308 -> 0.975
# SVC with param: {'C': 2, 'loss': 'hinge', 'penalty': 'l2', 'tol': 1e-08}

# import libraries
import sys
import os
import time
import csv
import re # regex
import time
import pandas as pd
import numpy as np

#from langdetect import detect 

from collections import Counter
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import ItalianStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold, ParameterGrid, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# import other files
from models.models import *
from utils.utils import *

# LOAD DATASETS
file_dev = './dataset/development.csv'
file_eval = './dataset/evaluation.csv'

file_dev = 'C:/Users/Latte/Documents/Politecnico/Data Science Lab/report_exam_winter_2020/Project Assignment/development.csv'
file_eval = 'C:/Users/Latte/Documents/Politecnico/Data Science Lab/report_exam_winter_2020/Project Assignment/evaluation.csv'


### development.csv
X_raw, y = loadData(file_dev,dev=True)
print(f"Shape X (raw) = {X_raw.shape}") # Shape X_raw = (28753, 1)
print(f"y = {len(y)}") # Shape y = (28753,)

### evaluation.csv
X_eval_raw = loadData(file_eval,dev=False)
print(f"Shape X (raw) = {X_eval_raw.shape}\n") # Shape X_eval_raw = (12323,1)

# count the items in each class
print(f"Element for each class: {dict(Counter(y))}") # {'pos': 19531, 'neg': 9222}

# recordPerClass=dict(Counter(y)) 
# print(recordPerClass) 
# print(f"Positive = {recordPerClass[1]/len(df_dev.index)*100:.2f} %")
# print(f"Negative = {recordPerClass[0]/len(df_dev.index)*100:.2f} %")

# UNBALANCED dataset:
# Positive = 67.93 %
# Negative = 32.07 %

# PREPROCESSING (data cleaning)
X_clean = preprocess(X_raw)
X_eval_clean = preprocess(X_eval_raw)
          
# added some non-sense words found printing samples
my_stopwords = ['lw','www', 'per', 'che', 'con', "engspa", 'zzzz', 'árco', 'área', 'âgée', 'http', 'www', 'per', 'che', 'con', 'era', 'del', 'della', 'tutto', 'come', 'ecc', 'etc', 'il','li','un','una', 'èesercito', 'ègrande', 'èil', 'èodificabile', 'èpoco', 'èra', 'élite', 'émotionnelle', 'équipe', 'était', 'étions', 'étoile', 'été']
whitelist = ['ma','stato','non','molto','solo','sono','contro']
stopwords = sw.words('italian') #+ my_stopwords 
          
# display top unigrams
show_top_unigrams(X_clean,20,show=False) #show=True print the list of top unigrams

## Class Stemmer and filter stopwords
class StemTokenizer(object): 
    def __init__(self): 
        self.stemmer = ItalianStemmer() 

    def __call__(self, document): 
        lemmas = []
        for t in word_tokenize(document, language='italian'): 
            t = t.strip() # leading whitespaces are eliminated
            lemma = self.stemmer.stem(t) # Stemmer
            # filter stopwords
            if t not in stopwords: #  and detect(t) == 'it' # to detect language
                if ( len(lemma) > 2 ) and ( len(lemma) < 16 ) :
                    lemmas.append(lemma)
            # allow words in the whitelit
            if t in whitelist :
                lemmas.append(lemma)
        return lemmas

stemmer = StemTokenizer()
#print(stemmer) #check

cv = CountVectorizer(encoding='utf-8',
                     lowercase=True, 
                     tokenizer=stemmer, 
                     stop_words=None, #filter stopwords in the stemmer
                     ngram_range=(1,2), 
                     max_df=1.0, #default
                     min_df=1, #default
                     # max_features=100000 #default
                    ) 

X = cv.fit_transform(X_clean)
print(X.shape)
X_eval = cv.transform(X_eval_clean)
print(X_eval.shape)
          
# TF-IDF
tr = TfidfTransformer()
X = tr.fit_transform(X)
X_eval = tr.fit_transform(X_eval)
# print(cv.vocabulary_.keys())
          
# DATA REDUCTION
# Incremental PCA
#from sklearn.decomposition import IncrementalPCA
#from sklearn.preprocessing import MaxAbsScaler
#ipca = IncrementalPCA(n_components=100)
#X_proj = ipca.fit_transform(X.toarray())

# DATA NORMALIZATION
# Scaler (Normalization)
#scaler = MaxAbsScaler()
#scaler.fit(X_proj)
#X_norm = scaler.transform(X_proj)
#X_eval_norm = scaler.transform(X_eval)
# data reduction -> worst solution

# k-fold (cross validation)
nfolds = 10

# select classifier (default: linearSVC=True, RandomForest=False, multinomialNB=False)
# in order to compute a parameter search set the corrisponding flag to True
clf = buildModel(X, y, nfolds, linearSVC=True, RandomForest=False, multinomialNB=False, parameter_search=False)

# Confusion Matrix
# display_confusion_matrix(clf, X, y, nfolds)

# PREDICT on evaluation.csv
clf.fit(X, y)
y_eval_pred = clf.predict(X_eval)

# save csv file 
saveFile(y_eval_pred)