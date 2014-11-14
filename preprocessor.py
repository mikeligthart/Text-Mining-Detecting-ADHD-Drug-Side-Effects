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
                
        self.template = template
        self.file = open(file_location, 'r')
        
        data = []
        for line in self.file.readlines():
            data.append(line.split(delimiter))
        self.data = pd.DataFrame(data, columns=template.headers)
        self.isProcessed = False

    def process(self, form='nltk'):
        self.transform_into_internal_format()

        if (form.lower() == 'nltk'):
            return self.transform_into_ntlk_format()
        else:
            return []

    def transform_into_internal_format(self):
        """The stored data is pre-processed feature by feature according to a given template.
           Features will be excluded, transformed to integers given a mapping or tokenized.
        """
        if (~self.isProcessed):
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
                    f = lambda x: int(x)
                    self.data[col] = self.data[col].apply(f)

                #Recode nominal data to integers
                elif(self.template.types[colindex].value == Datatype.dct.value):
                      self.data = self.data.replace({col:self.template.dicts[dictindex]})
                      dictindex += 1

                #Recode the labels to integers
                elif(self.template.types[colindex].value == Datatype.lbl.value):
                      self.data = self.data.replace({col:self.template.label})

                #Tokenize textual content
                elif(self.template.types[colindex].value == Datatype.con.value):
                    f = lambda x: nltk.word_tokenize(x)
                    self.data[col] = self.data[col].apply(f)            

                #Go to the next column
                colindex += 1
            self.isProcessed = True

    def transform_into_ntlk_format(self):
        label_name = Template().label_name
        labels = self.data[label_name].values.tolist()
        data = self.data.drop(label_name,1)
        data = data.to_dict('records')
        return list(zip(data,labels))      
    
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
