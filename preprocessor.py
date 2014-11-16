from enum import Enum
import numpy as np
import pandas as pd
import nltk


class Preprocessor(object):
    """Preprocessor splits batches of delimited data into seperate features according to a given template"""

    def __init__(self, file_location, delimiter, template):
        """The raw data will be read from a file,
           the items are seperated using a given delimiter and
           stored in a Pandas DataFrame"""
                
        self.file = open(file_location, 'r')
        self.template = template

        #Raw content will eventually be replaced with processed features.
        #For further analysis a copy of the raw content is stored.
        self.nested_content = []
        
        data = []
        for line in self.file.readlines():
            data.append(line.split(delimiter))
        self.data = pd.DataFrame(data, columns=template.headers)

        #Data only has to be processed once
        self.isProcessed = False

    def process(self, form='nltk', wordlist=[], degree=1):
        """Returns pre-processed data. The format can either be NLTK suitable or the internal format (Pandas.Dataframe).
           When building a trainingset the given wordlist is empty (default).
           When pre-processing test or real data the wordlist of the training set needs to be given.
        """
        
        if (~self.isProcessed):
            self.transform_into_internal_format(wordlist, degree)

        if (form.lower() == 'nltk'):
            return self.transform_into_ntlk_format()
        else:
            return self.data

    def transform_into_internal_format(self, wordlist, degree=1):
        """The stored data is pre-processed feature by feature according to a given template.
           Features will be excluded, transformed to integers given a mapping or tokenized.
        """
        colindex = 0
        dictindex = 0
        for col in self.template.headers:
            #Remove the features that need to be removed according to the template
            if(self.template.types[colindex].value == Datatype.rem.value): 
                self.data = self.data.drop(col,1)
                
            #Recode boolean data to integers
            elif(self.template.types[colindex].value == Datatype.bln.value): 
                self.data = self.data.replace({col:{'f':0,'t':1}})

            #Recode integer string to integers
            elif(self.template.types[colindex].value == Datatype.itg.value):
                to_int = lambda x: int(x)
                self.data[col] = self.data[col].apply(to_int)

            #Recode nominal data to integers
            elif(self.template.types[colindex].value == Datatype.dct.value):
                  self.data = self.data.replace({col:self.template.dicts[dictindex]})
                  dictindex += 1

            #Recode the labels to integers
            elif(self.template.types[colindex].value == Datatype.lbl.value):
                  self.data = self.data.replace({col:self.template.label})

            #Create new monogram features from the textual content.
            elif(self.template.types[colindex].value == Datatype.con.value):

                #Tokenize the textual content
                tokenize = lambda text: nltk.word_tokenize(text)
                self.data[col] = self.data[col].apply(tokenize)

                #Every words needs to be lowercase
                lower = lambda text: map(str.lower, text)
                self.data[col] = self.data[col].apply(lower)

                #Artifcats will be filtered from the data
                clean_artifcats = lambda text : [token.replace('\\n', '') for token in text]
                self.data[col] = self.data[col].apply(clean_artifcats)

                #If wordlist does not exist build a new one (building a training set)
                if not wordlist:
                    self.nested_content.append(self.aggregate_content_to_nested_list(self.data[col]))
                    self.wordlist = self.list_of_wordlists_to_ordered_wordlist(self.nested_content[-1], degree)
                #If wordlist exist we're dealing with unseen cases that needs to be mapped to the existing features
                else:
                    self.nested_content.append(self.aggregate_content_to_nested_list(self.data[col]))
                    self.wordlist = wordlist

                #Build monograms features from bags of words and collect frequencies per monogram for every record
                featurelist = self.generate_features_from_wordlists(self.wordlist, self.nested_content[-1])
                self.data = self.data.drop(col,1)
                self.data = pd.concat([self.data, featurelist], axis=1)

            #Go to the next column
            colindex += 1               
        self.isProcessed = True

    def transform_into_ntlk_format(self):
        """This method transforms the data to the NLTK format"""
        
        label_name = Template().label_name

        #Extract labels from table
        labels = self.data[label_name].values.tolist()
        data = self.data.drop(label_name,1)

        #Extract features from table
        data = data.to_dict('records')

        #Deliver (features,label) for every record in table in a list
        return list(zip(data,labels))

    def list_of_wordlists_to_ordered_wordlist(self, nested_content, degree=1):
        """This method builds a ordered wordlist from a list of nested content"""
        #Flatten nested_wordlist
        wordlist = [item for sublist in nested_content for item in sublist]

        if (degree > 1):
            wordlist = list(nltk.ngrams(wordlist,degree,pad_left=True, pad_right=False))

        #Order by frequency (from HF to LF)
        wordlist = nltk.FreqDist(wordlist).items()
        wordlist = sorted(wordlist, key=lambda tup: tup[1])[::-1]
        (wordlist, _) = zip(*wordlist)

        return wordlist

    def aggregate_content_to_nested_list(self, column):
        """This method nests all the seperate content blocks in a list"""
        nested_content = []

        #Extract bags of words from every entry in dataset
        extract_content = lambda content: nested_content.append(content)
        column.apply(extract_content)

        return nested_content

    def generate_features_from_wordlists(self, wordlist, nested_content, useFreq=False):
        """Every word in the wordlist will become a new features.
          Every word in the wordlist occurs 0 till n times in a content block.
          The values of the features are either represented as a boolean (default) or a frequency"""
        
        features = []

        for content in nested_content:
            freq_profile = []

            #Calculate the frequency of every word in a bag of words
            freqlist = nltk.FreqDist(content).items()
            (contentwords, freq) = zip(*freqlist)

            #For every new feature
            for word in wordlist:
                #Check if that occurs in the content
                if word in contentwords:
                    if useFreq:
                        #Yes-Yes: retrieve the freqency
                        freq_profile.append(freq[contentwords.index(word)])
                    else:
                        freq_profile.append(1)
                else:
                    #No: the frequency is 0
                    freq_profile.append(0)
            #Add the frequency profile for this record to the list of records
            features.append(pd.Series(freq_profile, wordlist))
            feature_table = pd.concat(features, axis=1)

        return feature_table.transpose()
            
                
    
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

class Template(object):
    """A template is used as a guide for preprocessing. It contains the names and types of the features (columns).
       Furthermore it containts a list of mappings from non-integer datatypes (e.g. boolean) and labels to integers.
       A template can also be used to a-priori exclude an irrelevant feature without deleting it from the dataset."""
                                              
    def __init__(self):
        self.label_name = 'label'
        self.headers = [] #Column/feature headers
        self.types = [] #Types of the features
        self.dicts = [] #List of mappings of nominal data to int. Ordered from left to right column-wise
        self.label = {} #Labels
