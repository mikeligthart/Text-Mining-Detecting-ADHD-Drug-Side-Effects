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
import random
import re

## SETTINGS ##
location = 'data/rivm/'
content_index = 9
label_index = 11
folds = os.listdir(location)
delimiter = '\t'
folds[:] = [location + i for i in folds]
positive_negative_ratio = 1.0

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
    # Retrive negative and positive portions of the data
    negative_data = [training_set[k][index] for index in range(0,len(training_set[k])) if training_labels[k][index] == 0]
    positive_data = [training_set[k][index] for index in range(0,len(training_set[k])) if training_labels[k][index] == 1]

    # Balancing techniques
    # 1. Upsampling: randomly copy positive data and inject that into the positive data set
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

feature_extractor = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords.words('dutch'))

## FEATURE SELECTION ##
feature_selector = LinearSVC(penalty="l1", dual=False)  

## CLASSIFIERS ##

# Naive Bayes
nb_clf = MultinomialNB()
nb_pip = Pipeline([('extractor', feature_extractor),('selector', feature_selector),('classifier', nb_clf)])
nb_pip.set_params(extractor__ngram_range = (1,1))

# Support Vector Machine
svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=100, class_weight = {1: 5})
svm_pip = Pipeline([('extractor', feature_extractor),('selector', feature_selector),('classifier', svm_clf)])
svm_pip.set_params(extractor__ngram_range = (1,1))

## EVALUATION - K-FOLD CROSS-VALIDATION - F1-SCORE ##
svm_f1_score = []
nb_f1_score = []
print "Start with %d-fold cross-validation" % len(folds)
for k in range(0,len(folds)):
    print "Fold %d" % k
    print "Evaluating SVM with standard balancing" 
    # Support Vector Machine
    svm_pip.fit(training_set[k], training_labels[k])
    svm_predicted_test_labels = svm_pip.predict(test_set[k])
    svm_f1_score.append(f1_score(test_labels[k], svm_predicted_test_labels))

    print "Evaluating NB with custom upsampling" 
    # Naive Bayes
    nb_pip.fit(upsampled_training_set[k], upsampled_training_labels[k])
    nb_predicted_test_labels = nb_pip.predict(test_set[k])
    nb_f1_score.append(f1_score(test_labels[k], nb_predicted_test_labels))

print "Avarage f1-score Support Vector Machine: %1.3f" % (sum(svm_f1_score)/len(svm_f1_score))
print "Avarate f1-score Naive Bayes: %1.3f" % (sum(nb_f1_score)/len(nb_f1_score))


