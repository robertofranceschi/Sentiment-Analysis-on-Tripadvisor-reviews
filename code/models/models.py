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


def loadData(file,dev=False) :       
    # check if files exist
    if not os.path.isfile(file) :
        print (f"File not found in specified path ({file})")
        sys.exit(1)
    df = pd.read_csv(file, encoding='utf-8')
    
    if dev : 
        ### development.csv
        print(f"Development set - n_records = {len(df.index)}\nShape dataframe = {df.values.shape}")
        
        # print(df.columns) # print column names - Index(['text', 'class'], dtype='object')
        df = df[~df['text'].isnull()] # Remove the rows where “Review” (column : 'text') is missing

        # df = removeNonItalianReviews(df) 
        # descriptive statistics summary (in data exploration)
        # summaryDataframe(df) 
          
        # convert from pos/neg to 1/0
        # class = { pos = 1, neg = 0 }
        labels = {'pos': 1,'neg': 0}  
        y = [labels[c] for c in df.values[:, -1]]
        return df.values[:, :-1], y
    else : 
        ### evaluation.csv
        print(f"\nEvaluation set - n_records = {len(df.index)}\nShape dataframe = {df.values.shape}")
        # print(df.columns) # print column names - Index(['text'], dtype='object')
        df = df[~df['text'].isnull()] # Remove the rows where “Review” (column : 'text') is missing
        return df.values[:]

def summaryDataframe(df) : 
    # data exploration
    # print a descriptive statistics summary of dataframe 'df'
    # by using .describe() function 
    
    print("Dataframe summary:")
    
    # preprocess 
    df['text'] = preprocess(df['text'])
    
    # Create new feature for the length of the review
    df['review_len'] = df['text'].astype(str).apply(len)
    
    # Create new feature for the word count of the review
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    # print results
    print(f"\nreview_len (n_chars) stats: \n{df['review_len'].describe()}")
    print(f"\nword x review stats: \n{df['word_count'].describe()}")
    return 
          
def removeNonItalianReviews(df_dev) : 
    # create a new colum in pandas dataframe and save the language of the review by using lang detect library
    df_dev['language'] = df_dev['text'].apply(lambda x: detect(x))
    df1 = pd.DataFrame(df_dev.groupby('language').text.count().sort_values(ascending=False))
    print(df1)
    print()
    print(df_dev.sample(10))
    print(len(df_dev))
    # remove lines in a foreign language
    indexNames = df_dev[ df_dev['language'] != 'it' ].index
    df_dev.drop(indexNames, inplace=True) 
    print(len(df_dev))
    
    return df_dev

# The distribution of top unigrams before/after removing stop words
def get_top_n_unigram(corpus, n):
    # remove 'stop_words' to appreciate the difference
    vec = CountVectorizer(ngram_range=(1,1), stop_words = stopwords).fit(corpus)
    unique_words = len(list(vec.get_feature_names()))
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    print(f"number of unique words = {unique_words}\n")
    return words_freq[:n]

def show_top_unigrams(X, n, show=False) : 
    common_words = get_top_n_unigram(X, n)
    if show : 
        print(f"Top {len(common_words)} UNIGRAMS:")
        for word, freq in common_words :
            print(word, freq)
    return common_words
          
def substituteEmoji(doc) :
    # this function allows to detect and substitute emoji with a default string (posemoji or negemoji)
    # it process one review at a time 
    emoji_dict = {'😢': 'negemoji ', '👎': 'negemoji ', '😒': 'negemoji ', '😖': 'negemoji ', '😠': 'negemoji ', '😡': 'negemoji ', '😤': 'negemoji ', '😨': 'negemoji ', '😱': 'negemoji ', '😳': 'negemoji ', '😬': 'negemoji ', '😞': 'negemoji ','🤐': 'negemoji ','😕': 'negemoji ','👍': 'posemoji ',
                  '😀': 'posemoji ', '💪': 'posemoji ','😎': 'posemoji ', '👌': 'posemoji ', '😁': 'posemoji ', '😃': 'posemoji ', '😃': 'posemoji ',  '😄': 'posemoji ', '😊': 'posemoji ', '😋': 'posemoji ', '😍': 'posemoji ', '🤗': 'posemoji ', '👏🏻': 'posemoji ', '😘': 'posemoji ', '🎉': 'posemoji ', '💗': 'posemoji ', '🔝': 'posemoji ', '😉': 'posemoji '}
    try:
        # UCS-4
        emoji = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        # UCS-2
        emoji = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]') 

    for specialEmoji in emoji_dict.keys():
        doc = doc.replace(specialEmoji, emoji_dict[specialEmoji])
    doc = emoji.sub(u' ', doc)
    return doc

def preprocess(review_text) :
    # data cleaning function
    X_clean = []
    for row in range(0, len(review_text)):       
        processed_review = str(review_text[row])
        # substitute emoji
        processed_review = substituteEmoji(processed_review)
        # replace \n
        processed_review = re.sub(r'\\n',' ',processed_review)
        # remove urls
        processed_review = re.sub(r'[hHtTpP]+[sS]?:[A-Za-z0-9-#_./]+',' ', processed_review)
        # replace _ (underscore)
        processed_review = re.sub(r'_',' ',processed_review)
        # remove all non alphabet letter
        processed_review = re.sub(r'[^a-zA-ZàáèéìíòóùúÀÁÈÉÌÍÙÚ]',' ', processed_review)
        # remove all the special characters
        processed_review = re.sub(r'\W', ' ', processed_review)
        # remove numbers
        processed_review = re.sub(r'\d+', ' ', processed_review) # [^a-zA-Z]      
        # remove all single characters
        processed_review = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_review)
        # Substituting multiple spaces with single space
        processed_review = re.sub(r'\s+', ' ', processed_review, flags=re.I)
        # Convert to lowercase
        processed_review = processed_review.lower()

        X_clean.append(processed_review)
    return X_clean

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

def buildModel(X, y, nfolds, linearSVC=True, RandomForest=False, multinomialNB=False, parameter_search=False) :
          
    # param_search=False -> run the classifier with the best configuration found until now
    # param_search=True -> run all the possible config of the parameterGrid
          
    if linearSVC : 
        return trainLinearSVC(X, y, nfolds, param_search=parameter_search)
    if RandomForest : 
        return trainRandomForest(X, y, nfolds, param_search=parameter_search)
    if multinomialNB : 
        return trainMultinomial(X, y, nfolds, comp=False, param_search=parameter_search) # to select complementarNB (comp=True)        
          
def display_confusion_matrix(clf, X, y, nfolds) :
    y_pred = cross_val_predict(clf, X, y, cv=nfolds)

    # Build the confusion matrix
    conf_mat = confusion_matrix(y, y_pred)
    print("\nConfusion matrix: ")
    print(conf_mat)

    # Plot the result in a prettier way
    conf_mat_df = pd.DataFrame(conf_mat) #, index = label_names, columns = label_names)
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'
    sns.heatmap(conf_mat_df, annot=False, cmap='GnBu', annot_kws={"size": 16}, fmt='g', cbar=False)
    plt.show()
    return 

def saveFile(y_pred) :
    # convert from 0/1 to neg/pos
    labels = ['neg','pos']
    y_print = [labels[c] for c in y_pred] 

    # write the file .csv
    path = 'c:/Users/robyf/Desktop/Politecnico/Data Science Lab/Python/LAB/Project Assignment/dataset/sub/'

    csvData = [['Id', 'Predicted']]
    for i in range (0, len(y_print)):
        csvData.append([i,y_print[i]])
    #print(csvData) 

    seq = (path,time.strftime("%d-%mh%H_%M"),".csv")
    filename = ''.join(seq)

    with open(filename, 'w', encoding='utf-8', newline='') as csvFile:
        wr = csv.writer(csvFile,delimiter=',')
        wr.writerows(csvData)
    csvFile.close()
    
    print(f'printed file in {filename}')
    return
