import nltk
import os
from enum import Enum

class Preprocessor3(object):

    def __init__(self, location, test_index, template, n_gram_degree=1, is_accumalative_n_gram=False):

        #Set class attributes
        self.template = template
        headers = self.template.headers
        types = self.template.types
        self.folds = os.listdir(location)
        self.folds[:] = [location + i for i in self.folds]

        #Build training set
        print('Start building training_set...')
        self.build_training_set(self.get_training_folds(self.folds, test_index), n_gram_degree, is_accumalative_n_gram)
        print('Finished building training_set...')
        
        #Build test set
        print('Start building test_set...')
        self.build_test_set(self.folds[test_index], n_gram_degree, is_accumalative_n_gram)
        print('Finished building test_set...')

    def build_training_set(self, folds, n_gram_degree, is_accumalative_n_gram):
        headers = list(self.template.headers)
        types = list(self.template.types)
        
        #Loading raw data
        raw_data = []
        for fold in folds:
            file = open(fold, 'r')
            for line in file.readlines():
                raw_data.append(line.split(self.template.delimiter))
            file.close()

        #Process raw data
        #1. obtain labels
        (labels, raw_data) = self.obtain_labels(raw_data, headers, types)
        print('1. Obtained labels')

        #2. removing features that are marked for removal
        remove_list = self.find_all_occurences_in_Datatype_list(Datatype.rem, types)
        raw_data = self.remove_features(raw_data, remove_list, headers, types)
        print('2. Removed unrequired features')
        
        #3. Clean content
        content_list = self.find_all_occurences_in_Datatype_list(Datatype.con, types)
        (raw_data, self.featureset, feature_list) = self.clear_content_and_initialize_featureset(raw_data, content_list)
        print('3. Cleared content')
        
        #4. Create n-grams and featuresets
        if n_gram_degree > 1:
            (raw_data, self.featureset, feature_list, headers, types) = self.create_ngrams_and_featuresets(raw_data, self.featureset, is_accumalative_n_gram, n_gram_degree, content_list, feature_list, headers, types)
        feature_start_index = len(raw_data[0])
        print('4. Created n-grams and initialized featuresets')
        
        #5. calculate feature values
        for content_index in range(0,len(content_list)):
            for feature_index in range(0,len(self.featureset[0])):
                (raw_data, headers, types) = self.calculate_feature_values(raw_data, self.featureset, feature_list, content_index, feature_index, headers, types)        
        print('5. Calculated feature values')
        
        #6. Rewrite to NLTK format
        #trainingset_data = []
        #for record_number in range(0,len(raw_data)):
            #trainingset_data.append(self.list_to_nltk_format(raw_data, feature_start_index, record_number, headers))
        #self.training_set = list(zip(trainingset_data, labels))
        self.training_set = raw_data
        print('6. Build training_set')

    def build_test_set(self, fold, n_gram_degree, is_accumalative_n_gram):
        headers = list(self.template.headers)
        types = list(self.template.types)
        
        #loading raw data
        raw_data = []
        file = open(fold, 'r')
        for line in file.readlines():
            raw_data.append(line.split(self.template.delimiter))
        file.close()

        #Process raw data
        #1. obtain labels
        (labels, raw_data) = self.obtain_labels(raw_data, headers, types)

        #2. removing features that are marked for removal
        remove_list = self.find_all_occurences_in_Datatype_list(Datatype.rem, types)
        raw_data = self.remove_features(raw_data, remove_list, headers, types)

        #3. Clean content
        content_list = self.find_all_occurences_in_Datatype_list(Datatype.con, types)
        (raw_data, _, feature_list) = self.clear_content_and_initialize_featureset(raw_data, content_list)

        #4. Create n-grams and featuresets
        if n_gram_degree > 1:
            (raw_data, _, feature_list, headers, types) = self.create_ngrams_and_featuresets(raw_data, self.featureset, is_accumalative_n_gram, n_gram_degree, content_list, feature_list, headers, types)
        feature_start_index = len(raw_data[0])

        #5. calculate feature values
        for content_index in range(0,len(content_list)):
            for feature_index in range(0,len(feature_list[0])):
                (raw_data, headers, types) = self.calculate_feature_values(raw_data, self.featureset, feature_list, content_index, feature_index, headers, types)

        #6. Rewrite to NLTK format
        #testset_data = []
        #for record_number in range(0,len(raw_data)):
            #testset_data.append(self.list_to_nltk_format(raw_data, feature_start_index, record_number, headers))
        #self.test_set = list(zip(testset_data, labels))
        self.test_set = raw_data
        
    ## PIPELINE METHODS ##
    #1.
    def obtain_labels(self, raw_data, headers, types):
        labels = []
        label_index = headers.index(self.template.label_name)
        for index in range(0,len(raw_data)):
            labels.append(raw_data[index][label_index])
            del raw_data[index][label_index]
        del headers[label_index]
        del types[label_index]
        return (labels, raw_data)

    #2.
    def remove_features(self, raw_data, remove_list, headers, types):
        remove_list.sort(reverse=True)
        for rem in remove_list:
            for index in range(0, len(raw_data)):
                del raw_data[index][rem]
            del headers[rem]
            del types[rem]
        return raw_data

    #3.
    #3.1
    def clear_content_and_initialize_featureset(self, raw_data, content_list):
        featureset =[[] for i in range(0, len(content_list))]
        feature_list = [[] for i in range(0, len(content_list))]
        content_index = 0
        for content_item in content_list:
            monograms = set()
            for index in range(0,len(raw_data)):          
                raw_data[index][content_item] = self.clean_content_box(raw_data[index][content_item])
                monograms = monograms | raw_data[index][content_item]
            featureset[content_index].append(list(monograms))
            feature_list[content_index].append(content_item)
            
            content_index += 1
        return (raw_data, featureset, feature_list)
    
    #3.2
    def clean_content_box(self, content):
        #3.2.1 Tokenize content into words
        content = nltk.regexp_tokenize(content, r'\w+')

        #3.2.2 Words to lower case
        content = [word.lower() for word in content]

        #3.2.3 Remove artifacts in content
        for artefact in self.template.artefacts:
            content = [word.replace(artefact,'') for word in content]

        #3.2.4 Remove non-relevant instances (word size < 4)
        content = [word for word in content if (len(word) >= 4)]
        #and word not in nltk.corpus.stopwords.words('dutch'))
        
        return set(content)

    #4.
    def create_ngrams_and_featuresets(self, raw_data, featureset, is_accumalative_n_gram, n_gram_degree, content_list, feature_list, headers, types):
        if is_accumalative_n_gram:
            for n in range(2,n_gram_degree+1):
                (raw_data, featureset, feature_list, headers, types) = self.create_ngram_per_n_and_featureset(raw_data, featureset, n, content_list, feature_list, headers, types)
        else:
            (raw_data, featureset, feature_list, headers, types) = self.create_ngram_per_n_and_featureset(raw_data, featureset, n_gram_degree, content_list, feature_list, headers, types)

        return (raw_data, featureset, feature_list, headers, types)

    def create_ngram_per_n_and_featureset(self, raw_data, featureset, n_gram_degree, content_list, feature_list, headers, types):
        content_index = 0
        for content_item in content_list:
            features = set()
            for index in range(0,len(raw_data)):                    
                ngrams = set(nltk.ngrams(raw_data[index][content_item], n_gram_degree))
                raw_data[index].append(ngrams)
                features = features | ngrams
            featureset[content_index].append(list(features))
            feature_list[content_index].append(len(raw_data[index])-1)
            headers.append(repr(n_gram_degree) + '_gram' + 'con_' + repr(content_item))
            types.append(Datatype.ngram)
            content_index += 1
        return (raw_data, featureset, feature_list, headers, types)

    #5.
    def calculate_feature_values(self, raw_data, featureset, feature_list, content_index, feature_index, headers, types):
        featureset_to_data_index = feature_list[content_index][feature_index]
        features = featureset[content_index][feature_index]
        for index in range(0, len(features)):
            for data_index in range(0, len(raw_data)):
                if features[index] in raw_data[data_index][featureset_to_data_index]:
                    raw_data[data_index].append(1)
                else:
                    raw_data[data_index].append(0)
            headers.append(features[index])
            types.append(Datatype.ngram_feature)

        return (raw_data, headers, types)

    #6.
    def list_to_nltk_format(self, raw_data, feature_start_index, record_number, headers):
        record = dict()
        for feature_number in range(feature_start_index,len(raw_data[record_number])):
            record[headers[feature_number]] = raw_data[record_number][feature_number]
        return record

    ## HELPER METHODS ##
    def find_all_occurences_in_Datatype_list(self, datatype, datatype_list):
        return [i for i, j in enumerate(datatype_list) if j.value == datatype.value]
      
    def get_training_folds(self, folds, index):
        if index == len(folds)-1:
            return folds[0:index]
        else:
            return folds[0:index] + folds[index+1:]

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
    ngram = 9#ngram
    ngram_feature = 10

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
