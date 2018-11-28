'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''


'''
~~~~~~~~ cat_accuracy ~~~~~~~~
Computes the accuracy of each category individually.
This is useful for exploratory analysis to determine categories that are suboptimal.  Such categories may need to be redefined.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
datacsv: .csv file as string, the raw csv containing the categorized item data.
train_size: float between 0 and 100, the percent of the total data set to be used as the training set.  Around 75% is recommended for our data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
accuracy_df: DataFrame, output data showing the category, the number of training examples, category size, and the accuracy of the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def cat_accuracy(datacsv, train_size):
    import pandas as pd
    import numpy as np
    from categorization import cat_data_prep
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    df, catdict = cat_data_prep(datacsv)

    # Compute a dictionary of all category keys and the number of items in each category.
    catsize_dict = dict.fromkeys(catdict)
    for key in catsize_dict:
        catsize_dict[key] = len(df[df['Category Key'] == key])

    # Generate training and testing sets using a numpy random sample in [0,1).
    msk = np.random.rand(len(df)) < train_size
    train_df = df[msk].reset_index()
    test_df = df[~msk].reset_index()

    X_train = train_df['Modified Description'].as_matrix()
    X_test = test_df['Modified Description'].as_matrix()
    y_train = train_df['Category Key'].as_matrix()
    y_test = test_df['Category Key'].as_matrix()

    # Run the model (time intensive)
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])
    text_clf.fit(X_train, y_train)
    y_test = text_clf.predict(X_test)

    # Using the category dictionary, create a column for the new category descriptions, and concat onto the dataframe.
    y_test_series = pd.Series(y_test).rename('Proposed Category Key')
    proposed_cats = []
    for key in y_test:
        proposed_cats.append(catdict[key])
    proposed_cats_series = pd.Series(proposed_cats).rename('Proposed Category Description')
    test_df2 = pd.concat([test_df, y_test_series, proposed_cats_series], axis=1)

    # Create new, unique series of categories and keys.
    unique_catnum_list = list(catdict.keys())
    unique_cat_list = []
    for i in unique_catnum_list:
        unique_cat_list.append(catdict[i])
    unique_catnum_series = pd.Series(unique_catnum_list).rename('Level 2 Key')
    unique_cat_series = pd.Series(unique_cat_list).rename('Level 2 Description')

    # Compute the category accuracies, along with the number of items in the category and the testing set size of that category.
    # Note that this is essentially what sklearn's train_test_split does, but done manually here.
    accuracy_scores = []
    category_size = []
    testing_size = []
    for key in unique_catnum_list:
        error_counter = 0
        temp_df = test_df2[test_df2['Category Key'] == key]
        working_df = pd.concat([temp_df['Category Key'], temp_df['Proposed Category Key']], axis=1)
        i = 0
        while i < len(working_df):
            if working_df['Category Key'].iloc[i] != working_df['Proposed Category Key'].iloc[i]:
                error_counter += 1
            i += 1
        if len(working_df) == 0:
            error_percent = 'No Data in Test Set'
        else:
            error_percent = (len(working_df) - error_counter) / len(working_df)
        accuracy_scores.append(error_percent)
        testing_size.append(len(working_df))
        category_size.append(catsize_dict[key])

    # Create series for each of the above computations, and append them to the final dataframe.
    accuracy_score_series = pd.Series(accuracy_scores).rename('Algorithm Accuracy')
    testing_size_series = pd.Series(testing_size).rename('Testing Set Size')
    category_size_series = pd.Series(category_size).rename('Total Category Size')
    accuracy_df = pd.concat([unique_catnum_series, unique_cat_series, accuracy_score_series, testing_size_series, category_size_series], axis=1)

    return(accuracy_df)
