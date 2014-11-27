import preprocessor
import rivm
import pickle

class Pipeline(object):

    def __init__(self, preprocessor, template):
        self.preprocessor = preprocessor
        self.template = template

    def preprocess(self, n_gram_degree, is_accumalative, data_location):

        data_save_location = data_location[0:len(data_location)-1] + '-processed/'
        
        print('#### Starting preprocessing #####')
        for index in range(0,self.template.number_of_folds):
            print('== Preprocessing fold ' + repr(index+1) + ' out of ' + repr(self.template.number_of_folds) + ' ==')
            preproc = preprocessor.process(data_location, index, self.template, n_gram_degree, is_accumalative)

            print('Saving training_set_' + repr(index) + '...')
            file_train = open(data_save_location + 'train' + repr(index) + '.pkl', 'wb')
            pickle.dump(preproc.training_set, file_train)
            file_train.close()

            file_train_label = open(data_save_location + 'train_label' + repr(index) + '.pkl', 'wb')
            pickle.dump(preproc.labels_train, file_train_label)
            file_train_label.close()
            print('Saved')

            print ('Savind header_' + repr(index) + '...')
            file_header = open(data_save_location + 'header' + repr(index) + '.pkl', 'wb')
            pickle.dump(preproc.headers, file_header)
            file_header.close()

            print('Saving test_set_' + repr(index) + '...')
            file_test = open(data_save_location + 'test' + repr(index) + '.pkl', 'wb')
            pickle.dump(preproc.test_set, file_test)
            file_test.close()

            file_test_label = open(data_save_location + 'test_label' + repr(index) + '.pkl', 'wb')
            pickle.dump(preproc.labels_test, file_test_label)
            file_test_label.close()
            print('Saved')
        print('#### Ppreprocessing finished #####')

    def analyse(self, file_location):

        #for index in range(0,self.template.number_of_folds):
        file = open('data/rivm-preprocessed/train0', 'rb')
        training_set_raw = pickle.load(file)
        file.close()

        file = open('data/rivm-preprocessed/train_label0', 'rb')
        training_labels = pickle.load(file)
        file.close()

        file = open('data/rivm-preprocessed/header0', 'rb')
        headers = pickle.load(file)
        file.close()
        
        file = open('data/rivm-preprocessed/test0', 'rb')
        test_set_raw = pickle.load(file)
        file.close()

        file = open('data/rivm-preprocessed/test_label0', 'rb')
        test_labels = pickle.load(file)
        file.close()
        
        #self.training_set = raw_to_nltk_format(training_set_raw, headers, training_label)
        #self.test_set = raw_to_nltk_format(test_set_raw, headers, test_label)

    ##Helper methods##
    def raw_to_nltk_format(self, raw, headers, labels):
        nltk_set = []
        for record_number in range(0, len(raw)):
            record = dict()
            for feature_number in range(0,len(raw[record_number])):
                record[headers[feature_number]] = raw[record_number][feature_number]
            nltk_set.add((record, labels[record_number]))
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
