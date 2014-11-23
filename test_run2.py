import preprocessor3
import rivm
import pickle

for index in range(0,10):
    preproc = preprocessor3.Preprocessor3('data/rivm/', index, rivm.RIVM_template(), 3, True)

    file_train = open('data/rivm-preprocessed/train' + index + '.pkl', 'wb')
    pickle.dump(preproc.training_set, file_train)
    file_train.close()

    file_test = open('data/rivm-preprocessed/test' +index + '.pkl', 'wb')
    pickle.dump(preproc.test_set, file_test)
    file_test.close()
