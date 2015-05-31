from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import regexp_tokenize
from nltk.stem.snowball import DutchStemmer
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
    
class Pipeline(object):

    def __init__(self, data, labels, resampler, verbose=True):
        self.data = data
        self.labels = labels
        self.verbose = verbose

        ## PIPELINE ELEMENTS ##
        # Feature extraction - FIXED #
        self.feature_extractor = TfidfVectorizer(tokenizer=self._tokenize, stop_words=stopwords.words('dutch'), ngram_range = (1,1))

        # Re-sampling - VARIABLE #
        self.resampler = resampler
    
        # Feature selection - FIXED #
        self.feature_selector = LinearSVC(penalty="l1", dual=False)

        # Classification - VARIABLE #
        self.nb = MultinomialNB()
        self.svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=100)

    def validation(self):
        ## K-FOLD CROSS-VALIDATION##
        f1_score_fold = [[],[]]

        if self.verbose:
            print "Start with %d-fold cross-validation" % len(self.data)
        for k in range(0, len(self.data)):
            #Creating test and training set per fold
            if self.verbose:
                print "Creating training en test set for fold %d" % (k+1) 
            training_set = [item for sublist in self.data[0:k] + self.data[k+1:len(self.data)]  for item in sublist]
            training_labels = [item for sublist in self.labels[0:k] + self.labels[k+1:len(self.data)] for item in sublist]
            test_set = self.data[k]
            test_labels = self.labels[k]

            #Feature extractions
            if self.verbose:
                print "Extracting features"
            training_set = self.feature_extractor.fit_transform(training_set)
            test_set = self.feature_extractor.transform(test_set)

            #Re-sampling
            if self.verbose:
                print "Re-sampling training set"
            training_set, training_labels = self.resampler.fit_transform(training_set, training_labels)

            #Feature selection
            if self.verbose:
                print "Selecting best features"
            training_set = self.feature_selector.fit_transform(training_set, training_labels)
            test_set = self.feature_selector.transform(test_set)

            #Training classifiers
            if self.verbose:
                print "Training classifiers"
            self.nb.fit(training_set, training_labels)
            self.svm.fit(training_set, training_labels)

            #Predicting class labels test set
            if self.verbose:
                print "Calculating f1-scores"
            nb_predicted_labels = self.nb.predict(test_set)
            svm_predicted_labels = self.svm.predict(test_set)

            f1_score_fold[0].append(f1_score(test_labels, nb_predicted_labels))
            f1_score_fold[1].append(f1_score(test_labels, svm_predicted_labels))

        ## RESULTS ##
        print "Avarage Naive Bayes f1_score is %f" % (sum(f1_score_fold[0])/len(f1_score_fold[0]))
        print "Avarage Support Vector Machine f1_score is %f" % (sum(f1_score_fold[1])/len(f1_score_fold[1]))
        return f1_score_fold

    def _tokenize(self, content):
        #Define Artefacts
        artefacts = ['\\n']
        quote = re.compile(r'quote.*(\\n\\n\\n|\\n\[\.\.\.\]\\n\\n|\n)')
        regexs = [quote]

        #Remove unwanted parts of text before tokenization
        for regex in regexs:
            content = regex.sub('', content)
        
        #Tokenize content into words
        content = regexp_tokenize(content, r'\w+')
        
        #Remove artifacts in content
        for artefact in artefacts:
            content = [word.replace(artefact,'') for word in content]

        #Stem words
        stemmer = DutchStemmer()
        content = [stemmer.stem(word) for word in content]
        return content
