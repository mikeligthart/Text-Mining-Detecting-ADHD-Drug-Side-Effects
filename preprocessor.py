import nltk
import os
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor(object):

    def __init__(self):
        self.training_set = []
        self.training_labels = []
        self.training_header = []
        self.test_set = []
        self.test_labels = []
        self.test_header = []
         
    def process(self, location, test_index, template, n_gram_degree=1, is_accumalative_n_gram=False, cut_off_freq=1, cut_off_max_size=1000):

        #Retrieve process data and settings
        self.folds = os.listdir(location)
        self.folds[:] = [location + i for i in self.folds]

        #Build training set
        print('Start building training_set.')
        (self.feature_sets, self.training_set, self.training_labels, self.training_header) = self.build_training_set(self.get_training_folds(self.folds, test_index), template, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size)
        print('Finished building training_set.')
        
        #Build test set
        print('Start building test_set.')
        (self.test_set, self.test_labels, self.test_header) = self.build_test_set(self.folds[test_index], template, self.feature_sets, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size)
        print('Finished building test_set.')

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
        print('1. Obtaining labels.')
        training_labels = self.obtain_labels(raw_data, template)
       
        #2. Process Raw Data into Training Set
        print('2. Start processing raw data into training set')
        #2.1 Process Content Blocks
        print('2.1 Start processing content blocks')
        content_list = self.find_all_occurences_in_Datatype_list(Datatype.content, template.types)
        #2.1.1 Create n-gram feature sets from content blocks
        print('2.1.1 Creating n-gram feature sets')
        (feature_sets, features_per_record) = self.create_n_gram_feature_sets(raw_data, content_list, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size)
        
        #2.1.2. calculate feature values
        print('2.1.2. Calculating content n-gram feature values')
        (content_feature_values, content_feature_header) = self.calculate_n_gram_feature_values(content_list, feature_sets, features_per_record)
        training_set += content_feature_values
        training_header += content_feature_header

        #2.2. Process Integers
        #2.3. Process Booleans
        #2.4. Process Standardized date and time
        #2.5. Process Dictionairy

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
        print('1. Obtaining labels.')
        test_labels = self.obtain_labels(raw_data, template)

        #2. Process Raw Data into Training Set
        print('2. Start processing raw data into test set')
        #2.1 Process Content Blocks
        print('2.1 Start processing content blocks')
        content_list = self.find_all_occurences_in_Datatype_list(Datatype.content, template.types) 
        #2.1.1 calculate features per record
        features_per_record = self.create_n_gram_features_per_record(raw_data, content_list, n_gram_degree, is_accumalative_n_gram)
        
        #2.1.2. calculate feature values and build training set
        print('2.1.2. Calculating content n-gram feature values')
        (content_feature_values, content_feature_header) = self.calculate_n_gram_feature_values(content_list, feature_sets, features_per_record)
        test_set += content_feature_values
        test_header += content_feature_header

        #2.2. Process Integers
        #2.3. Process Booleans
        #2.4. Process Standardized date and time
        #2.5. Process Dictionairy
        
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
    #2.1.1-A
    def create_n_gram_feature_sets(self, raw_data, content_list, n_gram_degree, is_accumalative_n_gram, cut_off_freq, cut_off_max_size):
        if is_accumalative_n_gram:
            complete_features_sets = [[[] for j in range(0,n_gram_degree)] for i in range(0, len(content_list))]
            vectorizer = TfidfVectorizer(tokenizer=Preprocessor.tokenize, stop_words=nltk.corpus.stopwords.words('dutch'), min_df=cut_off_freq, ngram_range=(1,n_gram_degree), max_features=cut_off_max_size)
        else:
            complete_features_sets = [[[]] for i in range(0, len(content_list))]
            vectorizer = TfidfVectorizer(tokenizer=Preprocessor.tokenize, stop_words=nltk.corpus.stopwords.words('dutch'), min_df=cut_off_freq, ngram_range=(n_gram_degree,n_gram_degree), max_features=cut_off_max_size)

        for content_index in range(0,len(content_list)):
            content_column = [row[content_list[content_index]] for row in raw_data]
            vectorizer.fit(content_column)
            if is_accumalative_n_gram:
                if n_gram_degree > 1:
                    mixed_feature_sets = vectorizer.get_feature_names()
                    for n in range(0, n_gram_degree):
                        ngrams = [ngram for ngram in mixed_feature_sets if ngram.count(' ') == n]
                        complete_features_sets[content_index][n] = ngrams
                else:
                    complete_features_sets[content_index][0] = vectorizer.get_feature_names()
            else:
                complete_features_sets[content_index][0] = vectorizer.get_feature_names()
            
            

        complete_features_per_record = self.create_n_gram_features_per_record(raw_data, content_list, n_gram_degree, is_accumalative_n_gram)
        return (complete_features_sets, complete_features_per_record)

    #2.1.1-B
    def create_n_gram_features_per_record(self, raw_data, content_list, n_gram_degree, is_accumalative_n_gram):        
        complete_features_per_record = [[] for i in range(0, len(content_list))]
        for content_index in range(0,len(content_list)):
            complete_features_per_record[content_index].append([])
            for data_index in range(0,len(raw_data)):          
                complete_features_per_record[content_index][0].append(Preprocessor.tokenize(raw_data[data_index][content_list[content_index]]))        

            if is_accumalative_n_gram:
                for n in range(2,n_gram_degree+1):
                    features_per_record = self.create_n_gram_feature_per_record_per_ngram(complete_features_per_record[content_index][0], n)
                    complete_features_per_record[content_index].append(features_per_record)
            else:
                if n_gram_degree > 1:
                        complete_features_per_record[content_index][0] = self.create_n_gram_feature_per_record_per_ngram(complete_features_per_record[content_index][0], n_gram_degree)
        self.complete_features_per_record = complete_features_per_record
        return complete_features_per_record

    #2.1.1-C
    def create_n_gram_feature_per_record_per_ngram(self, monograms_per_record, n_gram_degree):
        features_per_record = []
        for record in monograms_per_record:                    
            ngrams = list(nltk.ngrams(record, n_gram_degree))
            ngrams = list(map(Preprocessor.nltk_ngrams_to_nklearn, ngrams))
            features_per_record.append(ngrams)
        return features_per_record

    #2.1.2
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

    def nltk_ngrams_to_nklearn(tpl):
        result = tpl[0]
        for i in range(1,len(tpl)):
            result += ' '
            result += tpl[i]
            return result

    def tokenize(content):
        #Define Artefacts
        artefacts = ['\\n']
        
        #Tokenize content into words
        content = nltk.regexp_tokenize(content, r'\w+')

        #Remove artifacts in content
        for artefact in artefacts:
            content = [word.replace(artefact,'') for word in content]
    
        #Stem words
        stemmer = nltk.stem.snowball.DutchStemmer()
        content = [stemmer.stem(word) for word in content]

        return content

class Datatype(Enum):
    """Enum capturing the different types of data present in various datasets"""
                                              
    skip = 1 #Don't build features from these raw data elements
    label = 2 #label
    integer = 3 #integer
    boolean = 4 #boolean
    time = 5 #standardized date and time
    dictionairy = 6 #dictionairy
    content = 7 #textual content


