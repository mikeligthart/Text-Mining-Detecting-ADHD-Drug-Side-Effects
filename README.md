Text Mining: Detecting ADHD Drug Side Effects in Forum Posts
=============================
Source code to Ligthart, Mike, Text Mining: Detecting ADHD Drug Side Effects in Forum Posts, Text Mining Midterm Paper (LET-REMA-LCEX06), Radboud University, 2014

A. Generate results reported in paper
* Add RIVM data folds to folder '/data/rivm/'
* Run main.py - results will be outputted in the terminal

B. Explanation of the code.
The old code:

* old_main.py: calling the home build pipeline and collecting the scores
* pipeline.py: calling the preprocessor and training and testing (using analyser) the classifiers using 10-fold cross-validation
* analyser.py: collection class of the evaluation metrics: accuracy, precision, recall, f1-score
* preprocessor.py: large preprocessing class that tokenizes the input, calculates the td-idf feature values and returns a training and test set

New code: all the code is present in main.py expect for the tokenizer. That is taken from the old code (Preprocessor.tokenize).
