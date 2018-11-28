'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''


'''
~~~~~~~~ cat_data_prep_one ~~~~~~~~
Prepares only the categorized data for use with the SVM algorithm, assuming only one category is desired.
This function is useful for determining which of a new list of items belongs in a specific category.  This process results in improved algorithm accuracy, at the cost of flexibility.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
data: .csv filename as string, the csv containing the categorized item data.
supplier_delimiter: String with three options, replaces spaces in the supplier name with a delimiter.
    'underscore' - the algorithm vectorizer will treat each supplier name as one distinct term, while ignoring partial supplier names.
        Ex) "Medtronic" and "Medtronic Sofamor Danek" are considered two different supplier names.
    'space' - the vectorizer treats each word in the supplier name as distinct terms.  This will capture partial supplier names.
    None - default, the supplier names will not be used in the item descriptions.
level_2: String, the desired level 2 category to be found within the list of uncategorized items.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
finaldf: DataFrame, the categorized data in the correct format.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
def cat_data_prep(inputcsv, supplier_delimiter=None, level_2):
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

    # Create the category keys, where 1 denotes the desired category and 0 denotes other.
    catseries = catdf['Current ROi Contract Category Level 2']
    catlist = catseries.tolist()
    numlist = []
    for i in catlist:
        if i == level_2:
            numlist.append(1)
        else:
            numlist.append(0)
    cat_num_series = pd.Series(numlist).rename('Category Key')

    # Reset the index.
    df = df.reset_index(drop=True)

    # Concatenate the category keys to the final, prepped dataframe.
    finaldf = pd.concat([df, numseries], axis=1)

    # Print the number of unique items and the number of items in the desired category, for exploratory purposes.
    print("The total number of unique, categorized items is:" + str(len(finaldf['Primary Key'])))
    print("The total number of unique items in the desired category is:" + str(finaldf['Category Key'].sum()))

    return(finaldf, cat_dict)
