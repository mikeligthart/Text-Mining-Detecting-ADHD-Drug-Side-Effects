import pipeline
import rivm
import preprocessor
import time

start = time.time()
## Settings ##
data_location = 'data/rivm/'
n_gram_degree = 3
is_accumalative = True
cut_off_freq = 0.3
cut_off_max_size = 1500

## Loading instances ##
preproc = preprocessor.Preprocessor()
template = rivm.RIVM_template()
pipeline = pipeline.Pipeline(preproc, data_location, template)

## Run ##
#total_accuracy = []
#total_precision = []
#total_recall = [] #hehe
#total_f1 = []
(accuracy, precision, recall, f1, result_header) = pipeline.run(n_gram_degree, is_accumalative, cut_off_freq, cut_off_max_size)
print('time: ' + repr(time.time() - start) + ' s')
