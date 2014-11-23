import os
import nltk
import rivm
import preprocessor
from random import shuffle
import time

def get_train_folds(folds, index):
    if index == len(folds)-1:
        return folds[0:index]
    else:
        return folds[0:index] + folds[index+1:]
    
#Build training and test sets for n-folds crossvalidation
folds = os.listdir('data/rivm')
#shuffle(folds)
for index in range(0, len(folds)):
    preproc_train = preprocessor.Preprocessor('data/rivm', get_train_folds(folds, index),'\t',rivm.RIVM_template())
    preproc_train.process()
    preproc_train.save('data/rivm-preprocessed/train_' + str(index))
    preproc_test = preproc = preprocessor.Preprocessor('data/rivm', [folds[index]],'\t',rivm.RIVM_template())
    preproc_test.process(wordlist=preproc_train.wordlist)
    preproc_test.save('data/rivm-preprocessed/test_' + str(index))


    
