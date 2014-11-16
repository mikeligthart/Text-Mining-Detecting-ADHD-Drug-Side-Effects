import os
import nltk
import rivm
import preprocessor
from random import shuffle
import time

#Prepare folds
t_begin = time.time()
folds = os.listdir('data/rivm')
shuffle(folds)
preproc_train = preprocessor.Preprocessor('data/rivm', folds[0:8],'\t',rivm.RIVM_template())
train_set = preproc_train.process()
print(time.time() - t_begin)
preproc_test = preproc = preprocessor.Preprocessor('data/rivm', folds[9],'\t',rivm.RIVM_template())
test_set = preproc_test.process(wordlist=preproc_train.wordlist)
