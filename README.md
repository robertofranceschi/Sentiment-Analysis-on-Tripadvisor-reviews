# Sentiment Analysis on Tripadvisor reviews

Sentiment analysis, analyzing user’s textual reviews, to perform a binary classification task (i.e., understand if a comment includes a positive or negative mood). The dataset contains 41077 textual reviews specifically scraped from the tripadvisor.it website.

For all the detail about data exploration, data preprocessing, model selection and results [see the project report](./PDFs/report.pdf).

---
#### Models implemented
In the `models.py` has been implemented different classifiers: SVM (linear), RandomForest, Naive Bayes Classifiers (Multinomial). During training an hyperparameter search is performed.
In order to limit the effect of the unbalanced dataset (i.e. overfitting) for the training and validation phase a cross-validation approach was used, specifically with the implementation of Scikit-learn’s `KFold` class. The approach divides all the samples in groups of subsamples and consequently the prediction function was learned using k-1 folds and the fold left out was used for test. The default value for the number of folds was set to 10.


---

#### Results

The selection of the best model was done experimentally by using the different parameters, such as the number of features, the choice of tokenization (unigrams and bigrams), the use of a normalizer etc.
Overall the best accuracy 0.9694 was obtained using Linear SVM with stemmer and token unigrams plus bigrams as features and emoticons replacement as pre-processing technique. The main related issue was that with SVM was not performed aggressive feature selection.

---

This repo contains: 
- [Project Assignment](./PDFs/assignment.pdf)
- [Datasets](./datasets/)
- [code](./code/)
- [Project Report](./PDFs/report.pdf)
