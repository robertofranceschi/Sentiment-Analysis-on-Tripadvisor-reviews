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
# %matplotlib inline
import seaborn as sns


def buildModel(X, y, nfolds, linearSVC=True, RandomForest=False, multinomialNB=False, parameter_search=False) :
          
    # param_search = False -> run the classifier with the best configuration found until now
    # param_search = True  -> run all the possible config of the parameterGrid
          
    if linearSVC : 
        return trainLinearSVC(X, y, nfolds, param_search=parameter_search)
    if RandomForest : 
        return trainRandomForest(X, y, nfolds, param_search=parameter_search)
    if multinomialNB : 
        return trainMultinomial(X, y, nfolds, comp=False, param_search=parameter_search) # to select complementarNB (comp=True)        
    
def trainLinearSVC(X, y, nfolds = 10, param_search=False) :
    # Linear SVC (SVM) with ParameterGrid 
    best_config = {'C': 3, 'loss': 'hinge', 'penalty': 'l2', 'tol': 1e-08} # best configuration found
    start = time.time()
          
    if param_search : 
        params = {
            "penalty": ['l2'],
            "loss": ['squared_hinge','hinge'],
            "C": [0.5,2,3,5],
            "tol": [1e-08,1e-06,1e-05] # ,0.0001,0.001
        #   "max_iter": [1000,1500,2000]
        }
        print("Linear SVC with ParameterGrid -----")
        scores_list = [] # save the mean of scores (returned by cross_val_score) of different k-iteration
        for config in ParameterGrid(params) :
            print(f"\n### {config}")
            svc = LinearSVC(**config)
            scores = cross_val_score(svc, 
                                 X, # training data
                                 y, # training labels (class)
                                 scoring = 'f1_weighted', 
                                 cv = nfolds, 
                                 n_jobs=-1 # -1 = use all cores
                                )
            #print(scores)
            scores_list.append(scores.mean())
            print(f"F1-weighted : {scores.mean()}")
            print(f"Time Elapsed : {time.time()-start}")

        # choose the best configuration
        best_config = list(ParameterGrid(params))[np.argmax(scores_list)]
        print(f"\nBest config found {best_config}")

    # TRAINING and VALIDATION
    # run best config
    print("Linear SVC -----")
    svc = LinearSVC(**best_config)
    scores = cross_val_score(svc, 
                             X, # training data
                             y, # training labels (class)
                             scoring = 'f1_weighted', 
                             cv = nfolds, 
                             n_jobs=-1 # -1 = use all cores
                            )
    #print(scores) # print the np.array of different scores obtained with 
    print(f"F1-weighted : {scores.mean()}")
    print(f"Time Elapsed : {time.time()-start}\n")
    
    return svc

def trainRandomForest(X, y, nfolds = 10, param_search=False) : 
    # Random Forest Classifier
    best_config = {'n_estimators': 500, 'n_jobs' : -1} # best configuration found
          
    if param_search : 
        params = {
            'n_estimators': [100, 200, 400, 500, 1000, 2000], # number of decision trees
            'max_depth': [100, 200],
            'n_jobs' : [-1] # use all cores
        }

        print("Random Forest with ParameterGrid -----")
        scores_list = [] # save the mean of scores (returned by cross_val_score) of different k-iteration
        for config in ParameterGrid(params) :
            start = time.time()
            print(f"\n### {config}")
            clf = RandomForestClassifier(**config)
            scores = cross_val_score(clf, 
                                 X, # training data
                                 y, # training labels (class)
                                 scoring = 'f1_weighted', 
                                 cv = nfolds, 
                                 n_jobs=-1 # -1 = use all cores
                                )
            #print(scores)
            scores_list.append(scores.mean())
            print(f"F1-weighted : {scores.mean()}")
            print(f"Time Elapsed : {time.time()-start}")

        # choose the best configuration
        best_config = list(ParameterGrid(params))[np.argmax(scores_list)]
        print(f"\nBest config found {best_config}")

    # TRAINING and VALIDATION
    # run best config
    start = time.time()
    clf = RandomForestClassifier(**best_config)
    scores = cross_val_score(clf, 
                             X, # training data
                             y, # training labels (class)
                             scoring = 'f1_weighted', 
                             cv = nfolds, 
                             n_jobs=-1 # -1 = use all cores
                            )
    #print(scores) # print the np.array of different scores obtained with 
    print(f"F1-weighted : {scores.mean()}")
    print(f"Time Elapsed : {time.time()-start}\n")
    return clf

def trainMultinomial(X, y, nfolds, comp=False, param_search=False):
    if not comp :
        best_config = {'alpha' : 0 }
        if param_search : 
            params = {
                "alpha": [0,1e-06,1e-05,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,1,2,3,5], # smoothing parameter 
            }

            print("MultinomialNB with Parameter Grid -----")
            scores_list = []
            for config in ParameterGrid(params) :
                start = time.time()
                print(f"\n###{config}")
                multi = MultinomialNB(**config)
                scores = cross_val_score(multi, X, y, scoring='f1_weighted', cv=nfolds, n_jobs=-1)
                # print(scores)
                scores_list.append(scores.mean())
                print(f"F1-weighted : {scores.mean()}")
                print(f"Time Elapsed : {time.time()-start}\n")

            # choose the best configuration
            best_config = list(ParameterGrid(params))[np.argmax(scores_list)]
            print(f"Best config found {best_config}")

        # run best config
        print("MultinomialNB -----")
        start = time.time()
        multi = MultinomialNB(**best_config)
        scores = cross_val_score(multi, X, y, scoring='f1_weighted', cv=nfolds, n_jobs=-1)
        # print(scores)
        print(f"F1-weighted : {scores.mean()}")
        print(f"Time Elapsed : {time.time()-start}\n")
        return multi
    else : 
        # ComplementNB with ParameterGrid
        # The Complement Naive Bayes classifier was designed to correct the “severe assumptions” made by the 
        # standard Multinomial Naive Bayes classifier. It is particularly suited for imbalanced data sets.
        best_config = {'alpha' : 0 }
          
        if param_search : 
            params = {
                "alpha": [0,1e-06,1e-05,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,1,2,3,5],
            }

            print("ComplementNB with Parameter Grid -----")
            scores_list = []
            for config in ParameterGrid(params) :
                start = time.time()
                print(f"\n###{config}")
                cnb = ComplementNB(**config)
                scores = cross_val_score(cnb, X, y, scoring='f1_weighted', cv=nfolds, n_jobs=-1)
                #print(scores)
                scores_list.append(scores.mean())
                print(f"F1-weighted : {scores.mean()}")
                print(f"Time Elapsed : {time.time()-start}\n")

            # choose the best configuration
            best_config = list(ParameterGrid(params))[np.argmax(scores_list)]
            print(f"Best config found {best_config}")

        # run best config
        print("ComplementNB best -----")
        start = time.time()
        cnb = ComplementNB(**best_config)
        scores = cross_val_score(cnb, X, y, scoring='f1_weighted', cv=nfolds, n_jobs=-1)
        # print(scores)
        print(f"F1-weighted : {scores.mean()}")
        print(f"Time Elapsed : {time.time()-start}\n")
        return cnb
