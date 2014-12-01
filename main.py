import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from preprocessor import Preprocessor
import nltk
import random

## SETTINGS ##
location = 'data/rivm/'
content_index = 9
label_index = 11
number_of_folds = 10
folds = os.listdir(location)
delimiter = '\t'
folds[:] = [location + i for i in folds]

## LOAD RAW DATA ##
raw_data = []
for fold in folds:
    file = open(fold, 'r')
    for line in file.readlines():
        raw_data.append(line.split(delimiter))
    file.close()
data = [row[content_index] for row in raw_data]
labels = [row[label_index] for row in raw_data]
labels = [0 if label == 'f' else 1 for label in labels]

## BALANCE RAW DATA TO POSITIVE:NEGATIVR = 1:!
# Retrive negative and positive portions of the data
negative_data = [data[index] for index in range(0,len(data)) if labels[index] == 0]
positive_data = [data[index] for index in range(0,len(data)) if labels[index] == 1]

# Randomly copy positive data and inject that into the positive data set
positive_data_original_lenght = len(positive_data)
while(len(negative_data) > len(positive_data)):
    positive_data.append(positive_data[random.randint(0,positive_data_original_lenght)])   

# Create balanced data
balanced_data = []
balanced_labels = []
for i in range(0,(len(negative_data))):
    balanced_data.append(positive_data[i])
    balanced_labels.append(1)
    balanced_data.append(negative_data[i])
    balanced_labels.append(0)

# Randomize the order
mix = list(zip(balanced_data, balanced_labels))
random.shuffle(mix)
balanced_data, balanced_labels = zip(*mix)

## FEATURE EXTRACTION ##
feature_extractor = TfidfVectorizer(tokenizer=Preprocessor.tokenize, stop_words=nltk.corpus.stopwords.words('dutch'))

## FEATURE SELECTION ##
feature_selector = LinearSVC(penalty="l1", dual=False)  

## CLASSIFIERS ##
# Naive Bayes
nb_clf = MultinomialNB()
nb_pip = Pipeline([('extractor', feature_extractor),('selector', feature_selector),('classifier', nb_clf)])

# Support Vector Machine
svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)
svm_pip = Pipeline([('extractor', feature_extractor),('selector', feature_selector),('classifier', svm_clf)])

## EVALUATION USING GRID SEARCH ##
parameters = {'extractor__ngram_range': [(1,1), (2,2), (3,3), (1,2), (1,3), (2,3)]}

# Naive Bayes
nb_gs = GridSearchCV(nb_pip, parameters, 'f1',cv=number_of_folds, verbose=1)
nb_gs.fit(balanced_data, balanced_labels)
print(nb_gs.grid_scores_)

# Support Vector Machine
svm_gs = GridSearchCV(svm_pip, parameters, 'f1', cv=number_of_folds, verbose=1)
svm_gs.fit(balanced_data, balanced_labels)
print(svm_gs.grid_scores_)

