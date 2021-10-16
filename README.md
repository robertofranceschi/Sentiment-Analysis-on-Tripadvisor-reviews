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
```python
  from sklearn.metrics import f1_score
  f1_score(y_true, y_pred, average='weighted')
```

--- 

### 🗂 Folder organization
This repo is organized as follows: 
- `PDFs` contains details about experiments and discuss the results.
- `datasets` contains a local copy of the original dataset.
- `code` contains the different modules used to train and evaluate different models.

## 👨‍💻 Implementation 

Inside `models.py` are implemented different classifiers: 
- SVM (linear)
- RandomForest
- Näive Bayes Classifiers (Multinomial).

During training an hyperparameter search is performed (using a validation set).
In order to limit the effect of the unbalanced dataset (i.e. overfitting) for the training and validation phase a cross-validation approach was used, specifically with the implementation `KFold` class from `scikit-learn`. The approach divides all the samples in groups of subsamples and consequently the prediction function was learned using k-1 folds and the fold left out was used for test. The default value for the number of folds was set to 10.

The selection of the best model was done experimentally by using the different parameters and looking at the f1-score obtained on the validation set.

## Results

The f1-score (on test set) of the best performing model is **0.9694**. The model is a *Linear SVM* with stemmer, token unigrams plus bigrams as features and emoticons replacement as pre-processing technique.

▶ Further details about data exploration, data preprocessing, model selection and results [see the project report](./PDFs/report.pdf).

### References
[1] The original dataset can be downloaded [here](http://dbdmg.polito.it/wordpress/wp-content/uploads/2020/01/dataset_winter_2020.zip).
[2] Submission platform [link](http://35.158.140.217/)
