import nltk

class Analyser(object):
			
    def accuracy(classifier, test_set):
        return nltk.classify.accuracy(classifier, test_set)
    
    def precision(s1, s2):
        """
        Precision is the ratio of true positives (tp) to all predicted positives (tp + fp)
          s1: true labels
          s2: predicted labels
        """
        (_, tp, fp, _) = Analyser.prepare(s1, s2)
        if tp == 0 and fp == 0:
            return 0.0
        return 1.0 * tp / (tp + fp)

    def recall(s1, s2):
        """
        Recall is the ratio of true positives to all actual positives (tp + fn)
          s1: true labels
          s2: predicted labels
        """
        (_, tp, _, fn) = Analyser.prepare(s1, s2)
        if tp == 0 and fn == 0:
            return 0.0
        return 1.0 * tp / (tp + fn)

    def f1(s1, s2):
        """
        The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision (p) and recall (r).
          s1: true labels
          s2: predicted labels
        """
        p = Analyser.precision(s1, s2)
        r = Analyser.recall(s1, s2)
        if p == 0 and r == 0:
            return 0.0
        return 2.0 * p * r / (p + r)

    def prepare(s1, s2):
        tn = 0
        tp = 0
        fp = 0
        fn = 0
        for index in range(0, len(s1)):
            if s1[index] == s2[index]:
                if s1[index] == 'f':
                    tn += 1
                else:
                    tp += 1
            else:
                if s1[index] == 'f':
                    fp += 1
                else:
                    fn += 1
        return (tn, tp, fp, fn)
