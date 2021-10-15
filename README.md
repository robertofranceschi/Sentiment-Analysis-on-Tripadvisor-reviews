# Sentiment Analysis on Tripadvisor reviews 🛌🏽

## Problem Description

Perform sentiment analysis of textual reviews of hotel stays. The goal is to build a binary classifier able to understand whether the users expressed positive or negative feelings in their comments.

### Dataset

The dataset for this competition has been specifically scraped from the <tripadvisor.it> Italian web site. It contains 41077 textual reviews written in the Italian language.
The dataset is provided as textual files with multiple lines. Each line is composed of two fields: `text` and `class`. The `text` field contains the review written by the user, while the `class` field contains a label that can get the following values:
- **pos**: if the review shows a positive sentiment.
- **neg**: if the review shows a negative sentiment.

**Dataset tree hierarchy** The data have been distributed in two separate collections. Each collection is in a different file.
The dataset archive is organized as follows:
- `development.csv` (Development set): a collection of reviews **with** the class column. This collection of data has to be used during the development of the regression model.
- `evaluation.csv` (Evaluation set): a collection of reviews **without** the class column. This collection of data has to be used to produce the submission file.
- `sample_submission.csv`: a sample submission file.

### Evaluation metric
Your submissions will be evaluated exploiting the `f1_score` with the following configuration:
```
  from sklearn.metrics import f1_score
  f1_score(y_true, y_pred, average='weighted')
```

### 
This repo is organized as follows: 
- [Project Assignment](./PDFs/assignment.pdf)
- [Datasets](./datasets/)
- [code](./code/)
- [Project Report](./PDFs/report.pdf)

## Implementation 

In the `models.py` has been implemented different classifiers: SVM (linear), RandomForest, Naive Bayes Classifiers (Multinomial). During training an hyperparameter search is performed.
In order to limit the effect of the unbalanced dataset (i.e. overfitting) for the training and validation phase a cross-validation approach was used, specifically with the implementation of Scikit-learn’s `KFold` class. The approach divides all the samples in groups of subsamples and consequently the prediction function was learned using k-1 folds and the fold left out was used for test. The default value for the number of folds was set to 10.

The selection of the best model was done experimentally by using the different parameters, such as the number of features, the choice of tokenization (unigrams and bigrams), the use of a normalizer etc.

## Results

Overall the **best accuracy 0.9694** was obtained using *Linear SVM* with stemmer and token unigrams plus bigrams as features and emoticons replacement as pre-processing technique. The main related issue was that with SVM was not performed aggressive feature selection.

▶ Further details about data exploration, data preprocessing, model selection and results [see the project report](./PDFs/report.pdf).

### References
[1] Original dataset located [here](http://dbdmg.polito.it/wordpress/wp-content/uploads/2020/01/dataset_winter_2020.zip).
