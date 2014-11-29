import nltk
import pipeline
import os
from enum import Enum

class Preprocessor(object):

    def __init__(self):
        self.training_set = []
        self.training_labels = []
        self.training_header = []
        self.test_set = []
        self.test_labels = []
        self.test_header = []
         
    def process(self, location, test_index, template, n_gram_degree=1, is_accumalative_n_gram=False, cut_off_freq=2, cut_off_max_size=1000):

        #Retrieve process data and settings
        self.folds = os.listdir(location)
        self.folds[:] = [location + i for i in self.folds]

        #Build training set
        print('Start building training_set...')
        (feature_sets, self.training_set, self.training_labels, self.training_header) = self.build_training_set(self.get_training_folds(self.folds, test_index), template, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size)
        print('Finished building training_set...')
        
        #Build test set
        print('Start building test_set...')
        (self.test_set, self.test_labels, self.test_header) = self.build_test_set(self.folds[test_index], template, feature_sets, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size)
        print('Finished building test_set...')

    def build_training_set(self, folds, template, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size):      
        training_set = []
        training_header = []
        #Loading raw data
        raw_data = []
        for fold in folds:
            file = open(fold, 'r')
            for line in file.readlines():
                raw_data.append(line.split(template.delimiter))
            file.close()

        #Process raw data
        #1. obtain labels
        training_labels = self.obtain_labels(raw_data, template)
        print('1. Obtained labels')
        
        #2. Clean Raw Data
        print('2. Start cleaning raw data')
        #2.1 Clean Content Blocks
        content_list = self.find_all_occurences_in_Datatype_list(pipeline.Datatype.content, template.types)
        raw_data = self.clear_content(raw_data, content_list, template)
        print('2.1 Cleaned content blocks')

        #3. Process Raw Data into Training Set
        print('3. Start processing raw data into training set')
        #3.1 Process Content Blocks
        print('3.1 Start processing content blocks')
        #3.1.1 Create n-gram feature sets from content blocks
        (feature_sets, features_per_record) = self.create_n_gram_feature_sets(raw_data, content_list, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size)
        print('3.1.1 Created n-gram feature sets')
        
        #3.1.2. calculate feature values
        (content_feature_values, content_feature_header) = self.calculate_n_gram_feature_values(content_list, feature_sets, features_per_record)
        training_set += content_feature_values
        training_header += content_feature_header
        print('3.1.2. Calculated feature values and building of training set finished')

        #3.2. Process Integers
        #3.3. Process Booleans
        #3.4. Process Standardized date and time
        #3.5. Process Dictionairy

        return (feature_sets, training_set, training_labels, training_header)

    def build_test_set(self, fold, template, feature_sets, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size):
        test_set = []
        test_header = []

        #Loading raw data
        raw_data = []
        file = open(fold, 'r')
        for line in file.readlines():
            raw_data.append(line.split(template.delimiter))
        file.close()

        #Process raw data
        #1. obtain labels
        test_labels = self.obtain_labels(raw_data, template)
        print('1. Obtained labels')

        #2. Clean Raw Data
        print('2. Start cleaning raw data')
        #2.1 Clean Content Blocks
        content_list = self.find_all_occurences_in_Datatype_list(pipeline.Datatype.content, template.types)
        raw_data = self.clear_content(raw_data, content_list, template)
        print('2.1 Cleaned content blocks')

        #3. Process Raw Data into Training Set
        print('3. Start processing raw data into training set')
        #3.1 Process Content Blocks
        print('3.1 Start processing content blocks')
        #3.1.1 calculate features per record
        (_, features_per_record) = self.create_n_gram_feature_sets(raw_data, content_list, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size, True)
        
        #3.1.2. calculate feature values and build training set
        (content_feature_values, content_feature_header) = self.calculate_n_gram_feature_values(content_list, feature_sets, features_per_record)
        test_set += content_feature_values
        test_header += content_feature_header
        print('3.1.2. Calculated feature values and building of training set finished')

        #3.2. Process Integers
        #3.3. Process Booleans
        #3.4. Process Standardized date and time
        #3.5. Process Dictionairy
        
        return (test_set, test_labels, test_header)

    ## PIPELINE METHODS ##
    #1.
    def obtain_labels(self, raw_data, template):
        labels = []
        label_index = template.header.index(template.label_name)
        for index in range(0,len(raw_data)):
            labels.append(raw_data[index][label_index])
        return labels

    #2.
    #2.1-A
    def clear_content(self, raw_data, content_list, template):
        for content_index in range(0, len(content_list)):
            for index in range(0,len(raw_data)):          
                raw_data[index][content_list[content_index]] = self.clean_content_box(raw_data[index][content_list[content_index]], template)
        return raw_data
    #2.1-B
    def clean_content_box(self, content, template):
        #2.1-B.1 Tokenize content into words
        content = nltk.regexp_tokenize(content, r'\w+')

        #2.1-B.2 Remove artifacts in content
        for artefact in template.artefacts:
            content = [word.replace(artefact,'') for word in content]
    
        #2.1-B.3 Stem words
        stemmer = nltk.stem.snowball.DutchStemmer()
        content = [stemmer.stem(word) for word in content]

        #2.1-B.4 Remove stop words
        #content = [word for word in content if (len(word) >= 4)]
        content = [word for word in content if not word in set(nltk.corpus.stopwords.words('dutch'))]

        return content

    #3.
    #3.1.1-A
    def create_n_gram_feature_sets(self, raw_data, content_list, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size, is_for_test_set=False):
        complete_features_sets = [[] for i in range(0, len(content_list))]
        complete_features_per_record = [[] for i in range(0, len(content_list))]
        
        for content_index in range(0, len(content_list)):
            if is_accumalative_n_gram:
                for n in range(1,n_gram_degree+1):
                    (feature_sets, features_per_record) = self.create_n_gram_feature_set_per_ngram(raw_data, content_list[content_index], n, cut_off_freq, cut_off_max_size, is_for_test_set)
                    complete_features_sets[content_index].append(feature_sets)
                    complete_features_per_record[content_index].append(features_per_record)
            else:
                    (feature_sets, features_per_record) = self.create_n_gram_feature_set_per_ngram(raw_data, content_list[content_index], n_gram_degree, cut_off_freq, cut_off_max_size, is_for_test_set)
                    complete_features_sets[content_index].append(feature_sets)
                    complete_features_per_record[content_index].append(features_per_record)

        return (complete_features_sets, complete_features_per_record)

    #3.1.1-B
    def create_n_gram_feature_set_per_ngram(self, raw_data, content_item, n_gram_degree, cut_off_freq, cut_off_max_size, is_for_test_set):
        features = []
        features_per_record = []

        for index in range(0,len(raw_data)):                    
            ngrams = list(nltk.ngrams(raw_data[index][content_item], n_gram_degree))
            features += ngrams
            features_per_record.append(ngrams)

        if is_for_test_set:
            return (None, features_per_record)
        else:
            freq_features = {k:v for (k,v) in nltk.FreqDist(features).items() if v >= cut_off_freq}
            if len(freq_features) <= cut_off_max_size:
                features = list(freq_features.keys())
            else:
                features = list(freq_features.keys())[0:cut_off_max_size]
            return (features, features_per_record)

    

    #3.1.2
    def calculate_n_gram_feature_values(self, content_list, feature_sets, features_per_record):
        feature_values = [[[[[] for l in range(0,len(feature_sets[i][j]))] for k in range(0, len(features_per_record[i][j]))] for j in range(0, len(feature_sets[i]))] for i in range(0, len(content_list))]
        headers = [[[[] for l in range(0,len(feature_sets[i][j]))] for j in range(0, len(feature_sets[i]))] for i in range(0, len(content_list))]
        for content_index in range(0,len(content_list)):
            for n_gram_index in range(0,len(feature_sets[content_index])):
                word_index = 0
                for word in feature_sets[content_index][n_gram_index]:
                    for data_index in range(0, len(features_per_record[content_index][n_gram_index])):
                        if word in features_per_record[content_index][n_gram_index][data_index]:
                            feature_values[content_index][n_gram_index][data_index][word_index] = 1
                        else:
                            feature_values[content_index][n_gram_index][data_index][word_index] = 0
                    headers[content_index][n_gram_index][word_index] = word
                    word_index += 1
                        
        return (feature_values, headers)


    ## HELPER METHODS ##
    def find_all_occurences_in_Datatype_list(self, datatype, datatype_list):
        return [i for i, j in enumerate(datatype_list) if j.value == datatype.value]
      
    def get_training_folds(self, folds, index):
        if index == len(folds)-1:
            return folds[0:index]
        else:
            return folds[0:index] + folds[index+1:]

    ## STATIC METHODS ##
    def raw_to_nltk_format(feature_values, feature_header, labels):              
        nltk_set = []
        for record_number in range(0, len(feature_values[0][0])):        
            record = dict()
            for main_feature_category_index in range(0,len(feature_values)):
                for sub_feature_category_index in range(0, len(feature_values[main_feature_category_index])):
                    for feature_index in range(0, len(feature_values[main_feature_category_index][sub_feature_category_index][record_number])):
                        record[feature_header[main_feature_category_index][sub_feature_category_index][feature_index]] = feature_values[main_feature_category_index][sub_feature_category_index][record_number][feature_index]
            nltk_set.append((record, labels[record_number]))

        return nltk_set


