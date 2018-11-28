'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''


'''
~~~~~~~~ fixfinaldf ~~~~~~~~
Takes the previously uncategorized item data and reassigns the category text from the raw output determined by the SVM.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
uncatdf: DataFrame, the original uncategorized data.
y_test: Numpy Array, the learned output of the machine learning algorithm.
catdict: Dictionary, the list of categories and their keys generated in the "data_prep" functions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
uncatdf: DataFrame, the "final product" containing the previously uncategorized items and their new category descriptions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def fixfinaldf(uncatdf, y_test, catdict):
    import pandas as pd
    catdescriptions = []

    # Creates list of the category descriptions for each category key that was output from the learning algorithm.
    for key in y_test:
        catdescriptions.append(catdict[key])

    # Creates new columns on the previously uncategorized data corresponding to keys and descriptions.
    uncatdf['Level 2 Key'] = y_test
    uncatdf['Level 2 Description'] = catdescriptions

    return(uncatdf)
