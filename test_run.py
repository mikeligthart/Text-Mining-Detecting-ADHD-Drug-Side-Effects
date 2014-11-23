import nltk
import rivm
import preprocessor

preproc = preprocessor.Preprocessor('data/rivm/',['RIVM-ADHD.data.0.test'],'\t',rivm.RIVM_template())
featuresets = preproc.process()
#preproc.data.to_pickle('test.pkl')
#test_set, train_set = featuresets[190:], featuresets[:190]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier, test_set))
#print(classifier.show_most_informative_features(32))
