import preprocessor
import rivm
import time

begin = time.time()
preproc = preprocessor.Preprocessor()
preproc.process('data/rivm/', 5, rivm.RIVM_template(),3,True,1,1000)
print('time passed: ' + repr(time.time() - begin) + 's')
