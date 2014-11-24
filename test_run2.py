import preprocessor3
import rivm
import pickle

for index in range(0,10):
    print('== Starting with fold ' + repr(index+1) + ' out of 10 ==')
    preproc = preprocessor3.Preprocessor3('data/rivm/', index, rivm.RIVM_template(), 3, True)

    print('Saving training_set_' + repr(index) + '...')
    file_train = open('data/rivm-preprocessed/train' + repr(index) + '.pkl', 'wb')
    pickle.dump(preproc.training_set, file_train)
    file_train.close()
    print('Saved')

    print('Saving test_set_' + repr(index) + '...')
    file_test = open('data/rivm-preprocessed/test' + repr(index) + '.pkl', 'wb')
    pickle.dump(preproc.test_set, file_test)
    file_test.close()
    print('Saved')
