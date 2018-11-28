'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''


'''
~~~~~~~~ traintest ~~~~~~~~
Takes the prepared categorized data and performs a train-test split to determine algorithm accuracy.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
catdf: DataFrame, the prepped categorized data.
split_size: float, between 0 and 1.  Determines the percent of total data to be used as the testing set.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
score: float, the percent accuracy of the training-testing split.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def run_svm(catdf, split_size):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    # Convert data columns to np arrays for use with sklearn.
    X = np.array(catdf['Modified Description'])
    y = np.array(catdf['Category Key'])

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)

    # Set the model to vectorize the data, then fit.
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])
    text_clf.fit(X_train, y_train)
    # Print the accuracy of the trained model on the testing split.
    score = text_clf.score(X_test, y_test)
    print("The algorithm has computed the following average on this split:" + str(score))

    return(score)
