from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import regexp_tokenize
from nltk.stem.snowball import DutchStemmer
import re
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
    
class Pipeline(object):

    def __init__(self, data, labels, resampler, classifier, verbose=True):
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
        self.classifier = classifier

    def validation(self):
        ## K-FOLD CROSS-VALIDATION##
        training_set = []
        training_labels = []
        test_set = []
        test_labels = []
        predicted_labels = []
        f1_score_fold = []

        if self.verbose:
            print "Start with %d-fold cross-validation" % len(self.data)
        for k in range(0, len(self.data)):
            #Creating test and training set per fold
            if self.verbose:
                print "Creating training en test set for fold %d" % (k+1) 
            training_set.append([item for sublist in self.data[0:k] + self.data[k+1:len(self.data)]  for item in sublist])
            training_labels.append([item for sublist in self.labels[0:k] + self.labels[k+1:len(self.data)] for item in sublist])
            test_set.append(self.data[k])
            test_labels.append(self.labels[k])

            if self.verbose:
                print "Extracting features"
            #Feature extractions
            training_set[k] = self.feature_extractor.fit_transform(training_set[k])
            test_set[k] = self.feature_extractor.transform(test_set[k])

            if self.verbose:
                print "Re-sampling training set"
            #Re-sampling
            training_set[k], training_labels[k] = self.resampler.fit_transform(training_set[k], training_labels[k])

            if self.verbose:
                print "Selecting best features"
            #Feature selection
            training_set[k] = self.feature_selector.fit_transform(training_set[k], training_labels[k])
            test_set[k] = self.feature_selector.transform(test_set[k])

            if self.verbose:
                print "Training classifier"
            #Training classifiers
            self.classifier.fit_transform(training_set[k], training_labels[k])

            if self.verbose:
                print "Calculating f1-score"
            #Predicting class labels test set
            predicted_labels.append(self.classifier.predict(test_set[k]))
            f1_score_fold.append(f1_score(test_labels[k], predicted_labels[k]))

        ## RESULTS ##
        print "Avarage f1_score is %f" % (sum(f1_score_fold)/len(f1_score_fold))
        return (sum(f1_score_fold)/len(f1_score_fold)), f1_score_fold

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
