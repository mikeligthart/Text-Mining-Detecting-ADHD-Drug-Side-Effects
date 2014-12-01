ext Mining: Detecting ADHD Drug Side Effects in Forum Posts
=============================
Source code to Ligthart, Mike, Text Mining: Detecting ADHD Drug Side Effects in Forum Posts, Text Mining Midterm Paper (LET-REMA-LCEX06), Radboud University, 2014

A. Generate results reported in paper
1. Add RIVM folds to folder '/data/rivm/
2. Run main.py - results will be outputted to terminal

B. Explanation of the code.
1. The old code:
* old_main.py: calling the home build pipeline and collecting the scores
* pipeline.py: calling the preprocessor and training and testing (using analyser) the classifiers using 10-fold cross-validation
* analyser.py: collection class of the evaluation metrics: accuracy, precision, recall, f1-score
* preprocessor.py: large preprocessing class that tokenizes the input, calculates the td-idf feature values and returns a training and test set
2. New code: all the code is present in main.py expect for the tokenizer. That is taken from the old code (Preprocessor.tokenze).
