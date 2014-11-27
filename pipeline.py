import pickle
from enum import Enum
import time
import nltk
from analyser import Analyser

class Pipeline(object):

    def __init__(self, preprocessor, template):
        self.preprocessor = preprocessor
        self.template = template

    def preprocess(self, n_gram_degree, is_accumalative, data_location):

        self.data_save_location = data_location[0:len(data_location)-1] + '-preprocessed/'
        
        print('#### Starting preprocessing #####')
        for index in range(0,self.template.number_of_folds):
            print('== Preprocessing fold ' + repr(index+1) + ' out of ' + repr(self.template.number_of_folds) + ' ==')
            self.preprocessor.process(data_location, index, self.template, n_gram_degree, is_accumalative)

            print('Saving training_set_' + repr(index) + '...')
            file_train = open(self.data_save_location + 'train' + repr(index) + '.pkl', 'wb')
            pickle.dump(self.preprocessor.training_set, file_train)
            file_train.close()

            file_train_label = open(self.data_save_location + 'train_label' + repr(index) + '.pkl', 'wb')
            pickle.dump(self.preprocessor.labels_train, file_train_label)
            file_train_label.close()
            print('Saved')

            print ('Savind header_' + repr(index) + '...')
            file_header = open(self.data_save_location + 'header' + repr(index) + '.pkl', 'wb')
            pickle.dump(self.preprocessor.headers, file_header)
            file_header.close()

            print('Saving test_set_' + repr(index) + '...')
            file_test = open(self.data_save_location + 'test' + repr(index) + '.pkl', 'wb')
            pickle.dump(self.preprocessor.test_set, file_test)
            file_test.close()

            file_test_label = open(self.data_save_location + 'test_label' + repr(index) + '.pkl', 'wb')
            pickle.dump(self.preprocessor.labels_test, file_test_label)
            file_test_label.close()
            print('Saved')
        print('#### Ppreprocessing finished #####')

    def train_and_test_classifier(self, file_location):
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        print('loading files for analyses')
        #for index in range(0,self.template.number_of_folds):
        file = open(file_location + 'train0' + '.pkl', 'rb')
        training_set_raw = pickle.load(file)
        file.close()

        file = open(file_location + 'train_label0' + '.pkl', 'rb')
        training_labels = pickle.load(file)
        file.close()

        file = open(file_location + 'header0' + '.pkl', 'rb')
        headers = pickle.load(file)
        file.close()
        
        file = open(file_location + 'test0' + '.pkl', 'rb')
        test_set_raw = pickle.load(file)
        file.close()

        file = open(file_location + 'test_label0' + '.pkl', 'rb')
        test_labels = pickle.load(file)
        file.close()

        print('start building training set')
        self.training_set = self.raw_to_nltk_format(training_set_raw, headers, training_labels)

        print('start building test set')
        self.test_set = self.raw_to_nltk_format(test_set_raw, headers, test_labels)
        
        print('Train Naive Bayes Classifier')
        classifier = nltk.NaiveBayesClassifier.train(self.training_set)

        print('Calculate accuracy')
        self.accuracy.append(Analyser.accuracy(classifier, self.test_set))

        print('Calculate precision, recall and f1')
        (test_features, true_labels) = zip(*self.test_set)
        predicted_labels = classifier.classify_many(test_features)
        self.precision.append(Analyser.precision(true_labels, predicted_labels))
        self.recall.append(Analyser.recall(true_labels, predicted_labels))
        self.f1.append(Analyser.f1(true_labels, predicted_labels))
        
        
        
    ##Helper methods##
    def raw_to_nltk_format(self, raw, headers, labels):
        nltk_set = []
        for record_number in range(0, len(raw)):
            record = dict()
            for feature_number in range(0,len(raw[record_number])):
                record[headers[feature_number]] = raw[record_number][feature_number]
            nltk_set.append((record, labels[record_number]))
        return nltk_set


class Datatype(Enum):
    """Enum capturing the different types of data present in various datasets"""
                                              
    rem = 1 #remove
    bln = 2 #boolean
    itg = 3 #integer
    dct = 4 #dictionairy
    lbl = 5 #label
    zdt = 6 #standardized date and time
    sst = 7 #short string
    con = 8 #textual content
    ngram = 9 #ngram
    ngram_feature = 10 #ngram_feature

class Template(object):
    """A template is used as a guide for preprocessing. It contains the names and types of the features (columns).
       Furthermore it containts a list of mappings from non-integer datatypes (e.g. boolean) and labels to integers.
       A template can also be used to a-priori exclude an irrelevant feature without deleting it from the dataset."""
                                              
    def __init__(self):
        self.delimiter = [] #The delimiter
        self.label_name = 'class_label'
        self.headers = [] #Column/feature headers
        self.types = [] #Types of the features
        self.dicts = [] #List of mappings of nominal data to int. Ordered from left to right column-wise
        self.label = {} #Labels
        self.artefacts = []
        self.number_of_folds = 0
