import pipeline
import rivm
import preprocessor
import time

start = time.time()
## Settings ##
data_location = 'data/rivm/'
n_gram_degree = 1
is_accumalative = False
cut_off_freq = 2
cut_off_max_size = 1000

## Loading instances ##
preproc = preprocessor.Preprocessor()
template = rivm.RIVM_template()
pipeline = pipeline.Pipeline(preproc, data_location, template)

## Run ##
(accuracy, precision, recall, f1) = pipeline.run(n_gram_degree, is_accumalative, cut_off_freq, cut_off_max_size)
print('time: ' + repr(time.time() - start) + ' s')
