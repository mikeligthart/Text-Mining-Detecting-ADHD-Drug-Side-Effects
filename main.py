import pipeline
import rivm
import preprocessor
import time

preprocessor = preprocessor.Preprocessor()
template = rivm.RIVM_template()
#pipeline = pipeline.Pipeline(preprocessor, template)

#Preprocessing
#pipeline.preprocess(1, False, 'data/rivm/')

#Analysing
#pipeline.analyse(pipeline.data_save_location)
#pipeline.train_and_test_classifier('data/rivm-preprocessed/')

start = time.time()
preprocessor.process('data/rivm/', 9, template, 3, True)
print('time: ' + repr(time.time() - start) + ' s')
