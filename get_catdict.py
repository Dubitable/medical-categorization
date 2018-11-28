'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''


'''
~~~~~~~~ get_catdict ~~~~~~~~~
Obtains the category dictionary for the collection of categorized items.
Not necessary, but useful on the off chance that the dictionary granted from the "getdf" function is lost.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
df: DataFrame, the item data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
finaldf: DataFrame, the item data now with a column containing the category key.
cat_dict: Dictionary, the categories and their keys.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def get_catdict(df):

    # Same block of code as in the "data_prep" functions.  Generates a new column with the unique category keys, along with a reference dictionary.
    cat_list = df['Current ROi Contract Category Level 2'].tolist()
    cat_list_unique = list(set(cat_list))
    cat_keys = list(range(len(cat_list_unique)))
    cat_dict = dict.fromkeys(cat_keys)
    i = 0
    while i < len(cat_dict):
        cat_dict[i] = cat_list_unique[i]
        i += 1
    inverted_catdict = {v: k for k, v in cat_dict.items()}
    cat_num_list = []
    for cat in cat_list:
        cat_num_list.append(inverted_catdict[cat])
    cat_num_series = pd.Series(cat_num_list).rename('Category Key')

    # Reset the index and concatenate on the category key column.
    df = df.reset_index().iloc[:, 1:]
    finaldf = pd.concat([df, cat_num_series], axis=1)

    print("The total number of unique categories is:" + str(len(cat_dict)))

    return(finaldf, cat_dict)
