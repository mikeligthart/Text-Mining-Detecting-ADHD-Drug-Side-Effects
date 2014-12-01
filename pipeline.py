import nltk
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from analyser import Analyser
from preprocessor import Preprocessor


class Pipeline(object):

    def __init__(self, preprocessor, data_location, template):
        self.preprocessor = preprocessor
        self.data_location = data_location
        self.template = template

    def run(self, n_gram_degree=1, is_accumalative=False, cut_off_freq=1, cut_off_max_size=1000):
        result_header = ['C.1. Naive Bayes Classifier', 'C.2. Support Vector Machine', 'C.3. Decision Tree', 'C.4. k-NN']
        accuracy = [[] for n in range(0,len(result_header))]
        precision = [[] for n in range(0,len(result_header))]
        recall = [[] for n in range(0,len(result_header))]
        f1 = [[] for n in range(0,len(result_header))]
        
        print('**** Starting a ' + repr(self.template.number_of_folds) + '-fold cross validation analyses for findind the best classifier ****')
        for index in range(0,self.template.number_of_folds):
            print('== Preprocessing fold ' + repr(index+1) + ' out of ' + repr(self.template.number_of_folds) + ' ==')
            self.preprocessor.process(self.data_location, index, self.template, n_gram_degree, is_accumalative, cut_off_freq, cut_off_max_size)

            print('Training Classifiers')
            training_set = Preprocessor.raw_to_nltk_format(self.preprocessor.training_set, self.preprocessor.training_header, self.preprocessor.training_labels)
            print('C.1. training Naive Bayes Classifier')
            naive_bayes_classifier = nltk.NaiveBayesClassifier.train(training_set)
            print('C.2. training Support Vector Machine')
            svm_classifier = nltk.classify.scikitlearn.SklearnClassifier(LinearSVC()).train(training_set)
            print('C.3. training Decision Tree Classifier')
            decision_tree_classifier = nltk.DecisionTreeClassifier.train(training_set)
            print('C.4. training k-NN Classifier')
            knn_classifier = nltk.classify.scikitlearn.SklearnClassifier(KNeighborsClassifier()).train(training_set)

            print('Classifying Test Set')
            test_set = Preprocessor.raw_to_nltk_format(self.preprocessor.test_set, self.preprocessor.test_header, self.preprocessor.test_labels)
            (test_features, true_labels) = zip(*test_set)
            print('C.1. classifying with Naive Bayes')
            naive_bayes_predicted_labels = naive_bayes_classifier.classify_many(test_features)
            print('C.2. classifying with Support Vector Machine')
            svm_predicted_labels = svm_classifier.classify_many(test_features)
            print('C.3. classifying with Decision Tree Classifier')
            decision_tree_predicted_labels = decision_tree_classifier.classify_many(test_features)
            print('C.4. classifying k-NN')
            knn_predicted_labels = knn_classifier.classify_many(test_features)

            print('Calculating accuracy, precesion, recall and f1')
            print('C.1. Naive Bayes')
            accuracy[0].append(Analyser.accuracy(naive_bayes_classifier, test_set))       
            precision[0].append(Analyser.precision(true_labels, naive_bayes_predicted_labels))
            recall[0].append(Analyser.recall(true_labels, naive_bayes_predicted_labels))
            f1[0].append(Analyser.f1(true_labels, naive_bayes_predicted_labels))
            print('C.2. Support Vector Machine')
            accuracy[1].append(Analyser.accuracy(svm_classifier, test_set))       
            precision[1].append(Analyser.precision(true_labels, svm_predicted_labels))
            recall[1].append(Analyser.recall(true_labels, svm_predicted_labels))
            f1[1].append(Analyser.f1(true_labels, svm_predicted_labels))
            print('C.3. Decision Tree')
            accuracy[2].append(Analyser.accuracy(decision_tree_classifier, test_set))       
            precision[2].append(Analyser.precision(true_labels, decision_tree_predicted_labels))
            recall[2].append(Analyser.recall(true_labels, decision_tree_predicted_labels))
            f1[2].append(Analyser.f1(true_labels, decision_tree_predicted_labels))
            print('C.4. k-NN')
            accuracy[3].append(Analyser.accuracy(knn_classifier, test_set))       
            precision[3].append(Analyser.precision(true_labels, knn_predicted_labels))
            recall[3].append(Analyser.recall(true_labels, knn_predicted_labels))
            f1[3].append(Analyser.f1(true_labels, knn_predicted_labels))
            
        print('**** Finished with analyses *****')
        return(accuracy, precision, recall, f1, result_header)

class Template(object):
    """A template is used as a guide for preprocessing. It contains the names and types of the features (columns).
       Furthermore it containts a list of mappings from non-integer datatypes (e.g. boolean) and labels to integers.
       A template can also be used to a-priori exclude an irrelevant feature without deleting it from the dataset."""
                                              
    def __init__(self):
        self.delimiter = [] #The delimiter
        self.label_name = 'class_label'
        self.header = [] #Column/feature headers
        self.types = [] #Types of the features
        self.dicts = [] #List of mappings of nominal data to int. Ordered from left to right column-wise
        self.label = {} #Labels
        self.number_of_folds = 0
