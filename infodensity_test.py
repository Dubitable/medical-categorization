'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''

'''
~~~~~~~~ infodensity_test ~~~~~~~~
Computes the information density of each of the item descriptions.
The information density, by default, is the sum of the idf values divided by the length of the description.
Since higher idf values mean more rare words and thus more information, smaller descriptions with more descriptive terms will yeild higher info density values.
This is useful for determining overall how descriptive the short-texts are in your data set, useful for exploratory analysis.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
df: DataFrame, must be the data after it has been passed to any "data_prep" function.  May be either categorized or uncategorized.
normalized: Boolean, default False.  If True, normalizes the information density values with respect to the number of characters in the item description.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
final_df: DataFrame, the original df now with an additional column containing the information density of each item description.
id_mean: the average information density across all item descriptions.
id_stdev: the standard deviation of the information density across all item descriptions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def infodensity_test(df, normalized=False):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create an array from the modified descriptions.
    desc_series = df['Modified Description']
    desc_array = desc_series.as_matrix()

    # Pass the descriptions to the vectorizer and extract the vocublary and idf values.
    tfidf = TfidfVectorizer()
    tfidf.fit(desc_array)
    idfs = tfidf.idf_
    vocab = tfidf.vocabulary_
    idf_dict = dict(zip(vocab, idfs))
    documents = tfidf.inverse_transform(tfidf.transform(desc_array))

    # Compute the information density for each item description.
    id_values = []
    document_lens = []
    for document in documents:
        id_value = 0
        for term in document:
            id_value += idf_dict[term]
        if normalized == True:
            id_values.append(id_value / (len(document) ** 2))
        else:
            id_values.append(id_value / len(document))
        document_lens.append(len(document))

    # Compute the mean and standard deviation of the information density values across the entire corpus.
    id_mean = np.mean(id_values)
    id_stdev = np.std(id_values)

    # Create new columns in the dataframe for the newly calculated info density, along with the number of terms.
    id_df = pd.Series(id_values).rename('Information Density')
    doc_lens_df = pd.Series(document_lens).rename('Number of Terms')
    final_df = pd.concat([df, id_df, doc_lens_df], axis=1)

    return(final_df, id_mean, id_stdev)
