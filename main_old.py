import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk import regexp_tokenize
from nltk.stem.snowball import DutchStemmer
from unbalanced_dataset import SMOTE, UnderSampler
import random
import re

## SETTINGS ##
location = 'data/rivm/'
content_index = 9
label_index = 11
folds = os.listdir(location)
delimiter = '\t'
folds[:] = [location + i for i in folds]

## LOAD RAW DATA ##
raw_data = []
feat_content = []
labels = []
for fold_nr in range(0, len(folds)):
    file = open(folds[fold_nr], 'r')
    raw_data.append([])
    feat_content.append([])
    labels.append([])
    for line in file.readlines():
        raw_data[fold_nr].append(line.split(delimiter))
        feat_content[fold_nr] = [row[content_index] for row in raw_data[fold_nr]]
        labels[fold_nr] = [row[label_index] for row in raw_data[fold_nr]]
        labels[fold_nr] = [0 if label == 'f' else 1 for label in labels[fold_nr]]
    file.close()

## K-FOLD CROSS-VALIDATION with k = 10 ##
training_set = []
training_labels = []
test_set = []
test_labels = []

upsampled_training_set = []
upsampled_training_labels = []

for k in range(0, len(folds)):

    ## CREATING TRAINING AND TEST SET ##
    training_set.append([item for sublist in feat_content[0:k] + feat_content[k+1:len(folds)]  for item in sublist])
    training_labels.append([item for sublist in labels[0:k] + labels[k+1:len(folds)] for item in sublist])
    test_set.append(feat_content[k])
    test_labels.append(labels[k])

    ## BALANCING DATA ##
    # Balancing techniques
    # 1. Custum Upsampling: randomly copy positive data and inject that into the positive data set

    # Retrive negative and positive portions of the data
    negative_data = [training_set[k][index] for index in range(0,len(training_set[k])) if training_labels[k][index] == 0]
    positive_data = [training_set[k][index] for index in range(0,len(training_set[k])) if training_labels[k][index] == 1]
    
    positive_data_original_lenght = len(positive_data)
    while(len(positive_data)/float(len(negative_data)) < positive_negative_ratio):
        positive_data.append(positive_data[random.randint(0,positive_data_original_lenght-1)])   

    upsampled_training_set.append([])
    upsampled_training_labels.append([])
    for i in range(0,(len(positive_data))):
        upsampled_training_set[k].append(positive_data[i])
        upsampled_training_labels[k].append(1)

    for i in range(0,len(negative_data)):
        upsampled_training_set[k].append(negative_data[i])
        upsampled_training_labels[k].append(0)

    # Randomize the order
    mix = list(zip(upsampled_training_set[k], upsampled_training_labels[k]))
    random.shuffle(mix)
    upsampled_training_set[k], upsampled_training_labels[k] = zip(*mix)

# 2.
N_oversampling = SMOTE(verbose=True)

## FEATURE EXTRACTION ##
def tokenize(content):
    #Define Artefacts
    artefacts = ['\\n']
    quote = re.compile(r'quote.*(\\n\\n\\n|\\n\[\.\.\.\]\\n\\n|\n)')
    regexs = [quote]

    #Remove unwanted parts of text before tokenization
    for regex in regexs:
        content = regex.sub('', content)
    
    #Tokenize content into words
    content = regexp_tokenize(content, r'\w+')
    
    #Remove artifacts in content
    for artefact in artefacts:
        content = [word.replace(artefact,'') for word in content]

    #Stem words
    stemmer = DutchStemmer()
    content = [stemmer.stem(word) for word in content]
    return content

feature_extractor = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords.words('dutch'), ngram_range = (1,1))

## FEATURE SELECTION ##
feature_selector = LinearSVC(penalty="l1", dual=False)  

## CLASSIFIERS ##

# NAIVE BAYES
nb_clf = MultinomialNB()
nb_pip = Pipeline([('extractor', feature_extractor),('selector', feature_selector),('classifier', nb_clf)])

# SUPPORT VECTOR MACHINE
svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=100)

# Class weights
svm_weights_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=100, class_weight = {1: 5})
svm_weights_pip = Pipeline([('extractor', feature_extractor),('selector', feature_selector),('classifier', svm_weights_clf)])

# Nogueira oversampling
svm_N_oversampling_pip = Pipeline([('extractor', feature_extractor), ('re-sample', N_oversampling),('selector', feature_selector),('classifier', svm_clf)])

## EVALUATION - K-FOLD CROSS-VALIDATION - F1-SCORE ##
svm_weights_f1_score = []
svm_N_oversampling_f1_score = []
nb_f1_score = []
print "Start with %d-fold cross-validation" % len(folds)
for k in range(0,len(folds)):
    print "Fold %d" % k

    # Support Vector Machine
    """print "Evaluating SVM with class weights" 
    svm_weights_pip.fit(training_set[k], training_labels[k])
    svm_predicted_test_labels = svm_weights_pip.predict(test_set[k])
    svm_weights_f1_score.append(f1_score(test_labels[k], svm_predicted_test_labels))"""

    print "Evaluating SVM with Nogueira oversampling" 
    svm_N_oversampling_pip.fit(training_set[k], training_labels[k])
    svm_predicted_test_labels = svm_N_oversampling_pip.predict(test_set[k])
    svm_N_oversampling_f1_score.append(f1_score(test_labels[k], svm_predicted_test_labels))

    # Naive Bayes
    """print "Evaluating NB with custom upsampling" 
    nb_pip.fit(upsampled_training_set[k], upsampled_training_labels[k])
    nb_predicted_test_labels = nb_pip.predict(test_set[k])
    nb_f1_score.append(f1_score(test_labels[k], nb_predicted_test_labels))"""

#print "Avarage f1-score Support Vector Machine with class weights: %1.3f" % (sum(svm_weights_f1_score)/len(svm_weights_f1_score))
print "Avarage f1-score Support Vector Machine with Nogueira oversampling: %1.3f" % (sum(svm_N_oversampling_f1_score)/len(svm_N_oversampling_f1_score))
#print "Avarate f1-score Naive Bayes: %1.3f" % (sum(nb_f1_score)/len(nb_f1_score))


