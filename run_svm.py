'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''


'''
~~~~~~~~ run_svm ~~~~~~~~
Takes the prepared data, computes the model, and predicts the new categories.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
catdf: DataFrame, the prepped categorized data.
uncatdf: DataFrame, the prepped uncategorized data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
y_test: np array, the predicted category keys for the uncategorized data, ordered.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def run_svm(catdf, uncatdf):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    # Convert data columns to np arrays for use with sklearn.
    X_train = np.array(catdf['Modified Description'])
    y_train = np.array(catdf['Category Key'])
    X_test = np.array(uncatdf['Modified Description'])

    # Set the model to vectorize the data, then fit to the categorized data.
    # Note that a GridSearchCV was performed to tune the parameters of the vectorizer and SVM, however the accuracy gains were minimal, if any.
    # If the training dataset is drastically changed, it is advised to repeat the process in case the best parameters have changed.
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])
    text_clf.fit(X_train, y_train)
    # Use the fitted model to predict the categories for the uncategorized data.
    y_test = text_clf.predict(X_test)

    return(y_test)
