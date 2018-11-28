'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''

'''
~~~~~~~~ pre_process ~~~~~~~~
Implements two preprocessing procedures:
    1) Removes items with suppliers that have less than a certain number of total items.
        - These suppliers typically have very poor data quality among other problems.
    2) Removes items with insufficient item descriptions.
        - These item descriptions have insufficient accuracy with the algorithm.  It is best to categorize these manually.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Definitions:
idf: the inverse document frequency of the term.  In theory, words that are more unique throught all documents encapsulate more information.
    Thus, a higher idf value is more useful than a low idf value.
Integrity Value: the sum of all idf values for all terms in an item description.  This yeilds a reliable metric for quality of the description.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
catdf: DataFrame, the categorized item data obtained from one of the "data_prep" scripts.
supplier_count_threshold: int, default 1, the number of items a supplier must have in order to be kept in the data set.
    ie) if Stryker has 4 items but Medtronic has 2, then if supplier_count_threshold=2, all of the Medtronic items are thrown out.
int_thresh: float, default False.  Sets the integrity value threshold to a specific number.  If left false, the integrity value threshold is generated normally.
int_std_scale: float, default 1.  Sets the number of standard deviations outside the mean of the integrity value to exclude.
    ie) 1 means anything further than one standard deviation from the mean is excluded, 2 excludes everything outside 2 stdevs, etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
catpassdf: DataFrame, the categorized items that passed both preprocessing thresholds.
catfaildf: DataFrame, the categorized items that did not pass the thresholds.
integrity_thresh: float, the final threshold used to exclude items with poor descriptions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def pre_process(catdf, supplier_count_threshold=1, int_thresh=False, int_std_scale=1):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    #   1) Remove any supplier with less than x items in total.
    supplier_counts = catdf['ROi Supplier Name'].value_counts()
    supplier_fail = supplier_counts[supplier_counts <= supplier_count_threshold].index.tolist()
    # Create a mask based on if the supplier name fails to have a high enough number of items, and create a holding dataframe for the failed items.
    msk = catdf['ROi Supplier Name'].isin(supplier_fail)
    supplier_fail_df = catdf[msk].reset_index(drop=True)
    # Take only the items that were not in the above mask.
    catdf = catdf[~msk].reset_index(drop=True)
    print('Number of suppliers removed due to having too few items in training set:' + str(len(supplier_fail)))
    print('Number of items removed due to supplier having too few items in training set:' + str(len(supplier_fail_df)))

    #   2) Generate a threshold for the amount of information contained by an item description.
    cat_desc = catdf['Modified Description'].as_matrix()

    # Vectorize the item descriptions using the TfidfVectorizer.
    cat_tfidf = TfidfVectorizer()
    cat_tfidf.fit(cat_desc)

    # Extract the vocabulary of the corpus and the individual idf values for each term in each document (item description).
    cat_idfs = cat_tfidf.idf_
    cat_vocab = cat_tfidf.vocabulary_
    cat_idf_dict = dict(zip(cat_vocab, cat_idfs))
    cat_documents = cat_tfidf.inverse_transform(cat_tfidf.transform(cat_desc))

    # Create a list of all integrity values for each item description.
    cat_ivs = []
    for document in cat_documents:
        cat_iv = 0
        for term in document:
            cat_iv += cat_idf_dict[term]
        cat_ivs.append(cat_iv)

    # Find the mean and standard deviation of all item descriptions' integrity values.  This will be used for the threshold unless stated otherwise.
    integrity_mean = np.mean(cat_ivs)
    integrity_std = np.std(cat_ivs)
    if int_thresh != False:
        integrity_thresh = int_thresh
    else:
        integrity_thresh = integrity_mean - (int_std_scale * integrity_std)

    # Filter the items based on wheather they pass or fail the integrity value threshold.
    cat_ivs_series = pd.Series(cat_ivs).rename('Integrity Value')
    catdf = pd.concat([catdf, cat_ivs_series], axis=1)
    cat_pass_df = catdf[catdf['Integrity Value'] > integrity_thresh].reset_index(drop=True)
    cat_fail_df = catdf[catdf['Integrity Value'] <= integrity_thresh].reset_index(drop=True)

    print('Number of items removed due to low integrity item descriptions:' + str(len(cat_fail_df)))

    return(cat_pass_df, cat_fail_df, integrity_thresh)
