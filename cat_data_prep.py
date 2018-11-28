'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''

'''
~~~~~~~~ cat_data_prep ~~~~~~~~
Prepares only the categorized data for use with the SVM algorithm.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
inputcsv: .csv filename as string, the raw data file containing all categorized item data.
supplier_delimiter: String with three options, replaces spaces in the supplier name with a delimiter.
    'underscore' - the algorithm vectorizer will treat each supplier name as one distinct term, while ignoring partial supplier names.
        Ex) "Medtronic" and "Medtronic Sofamor Danek" are considered two different supplier names.
    'space' - the vectorizer treats each word in the supplier name as distinct terms.  This will capture partial supplier names.
    None - default, the supplier names will not be used in the item descriptions.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
finaldf: DataFrame, the categorized data ready for use with the algorithm.
cat_dict: Dictionary, the categories and their keys.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def cat_data_prep(inputcsv, supplier_delimiter=None):
    import pandas as pd

    df = pd.read_csv(inputcsv, encoding='latin1')

    # Creates primary key from the supplier name and item number.  Also creates modified description by potentially concatenating supplier name to the item description.
    if supplier_delimiter == 'underscore':
        primary_key = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Supplier Item Number']).rename('Primary Key')
        modified_description = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Item Description']).rename('Modified Description')
    elif supplier_delimiter == 'space':
        primary_key = (df['ROi Supplier Name'] + " " + df['ROi Supplier Item Number']).rename('Primary Key')
        modified_description = (df['ROi Supplier Name'] + " " + df['ROi Item Description']).rename('Modified Description')
    elif supplier_delimiter == None:
        primary_key = (df['ROi Supplier Name'] + " " + df['ROi Supplier Item Number']).rename('Primary Key')
        modified_description = df['ROi Item Description'].rename('Modified Description')
    df = pd.concat([primary_key, modified_description, df['ROi Supplier Name'], df['ROi Supplier Item Number'], df['ROi Item Description'], df['Current ROi Contract Category Level 2']], axis=1)

    # Drop any nulls or duplicates.
    df = df.dropna(how='any', axis=0)
    df = df.drop_duplicates(subset=['Primary Key'], keep='first')

    # Create the category dictionary for use later. This is necessary because we must pass the categories as integers to the sklearn algorithm.
    cat_series = df['Current ROi Contract Category Level 2']
    cat_list = cat_series.tolist()
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

    # Reset the index.
    df = df.reset_index(drop=True)

    # Concatenate the category keys to the final, prepped dataframe.
    finaldf = pd.concat([df, cat_num_series], axis=1)

    # Print the number of unique items and categories, for exploratory purposes.
    print("The total number of unique, categorized items is:" + str(len(finaldf['Primary Key'])))
    print("The total number of unique categories is:" + str(len(cat_dict)))

    return(finaldf, cat_dict)