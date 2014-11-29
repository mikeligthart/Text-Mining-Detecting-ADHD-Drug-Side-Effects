import pickle
from enum import Enum
import time
import nltk
from analyser import Analyser
from preprocessor import Preprocessor

class Pipeline(object):

    def __init__(self, preprocessor, data_location, template):
        self.preprocessor = preprocessor
        self.data_location = data_location
        self.template = template

    def run(self, n_gram_degree=1, is_accumalative=False, cut_off_freq=2, cut_off_max_size=1000):
        accuracy = []
        precision = []
        recall = []
        f1 = []
        for index in range(0,self.template.number_of_folds):
            print('== Preprocessing fold ' + repr(index+1) + ' out of ' + repr(self.template.number_of_folds) + ' ==')
            self.preprocessor.process(self.data_location, index, self.template, n_gram_degree, is_accumalative, cut_off_freq, cut_off_max_size)

            print('Training Classifiers')
            training_set = Preprocessor.raw_to_nltk_format(self.preprocessor.training_set, self.preprocessor.training_header, self.preprocessor.training_labels)
            print('C.1. training Naive Bayes Classifier')
            naive_bayes_classifier = nltk.NaiveBayesClassifier.train(training_set)
            #print('C.2. training ')
            #nltk.WekaClassifier.train(

            print('Classifying Test Set')
            test_set = Preprocessor.raw_to_nltk_format(self.preprocessor.test_set, self.preprocessor.test_header, self.preprocessor.test_labels)
            (test_features, true_labels) = zip(*test_set)
            print('C.1. classifying with Naive Bayes')
            naive_bayes_predicted_labels = naive_bayes_classifier.classify_many(test_features)

            print('Calculating accuracy, precesion, recall and f1')
            print('Naive Bayes')
            accuracy.append(Analyser.accuracy(naive_bayes_classifier, test_set))       
            precision.append(Analyser.precision(true_labels, naive_bayes_predicted_labels))
            recall.append(Analyser.recall(true_labels, naive_bayes_predicted_labels))
            f1.append(Analyser.f1(true_labels, naive_bayes_predicted_labels)) 

        return(accuracy, precision, recall, f1)

class Datatype(Enum):
    """Enum capturing the different types of data present in various datasets"""
                                              
    skip = 1 #Don't build features from these raw data elements
    label = 2 #label
    integer = 3 #integer
    boolean = 4 #boolean
    time = 5 #standardized date and time
    dictionairy = 6 #dictionairy
    content = 7 #textual content

class Template(object):
    """A template is used as a guide for preprocessing. It contains the names and types of the features (columns).
       Furthermore it containts a list of mappings from non-integer datatypes (e.g. boolean) and labels to integers.
       A template can also be used to a-priori exclude an irrelevant feature without deleting it from the dataset."""
                                              
    def __init__(self):
        self.delimiter = [] #The delimiter
        self.label_name = 'class_label'
        self.header = [] #Column/feature headers
        self.types = [] #Types of the features
        self.dicts = [] #List of mappings of nominal data to int. Ordered from left to right column-wise
        self.label = {} #Labels
        self.artefacts = []
        self.number_of_folds = 0
