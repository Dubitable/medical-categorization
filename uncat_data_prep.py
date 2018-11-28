'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''


'''
~~~~~~~~ uncat_data_prep ~~~~~~~~
Prepares only the uncategorized data for use with the SVM algorithm.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
inputcsv: .csv filename as string, the raw data file containing all uncategorized item data.
supplier_delimiter: String with three options, replaces spaces in the supplier name with a delimiter.
    'underscore' - the algorithm vectorizer will treat each supplier name as one distinct term, while ignoring partial supplier names.
        Ex) "Medtronic" and "Medtronic Sofamor Danek" are considered two different supplier names.
    'space' - the vectorizer treats each word in the supplier name as distinct terms.  This will capture partial supplier names.
    None - default, the supplier names will not be used in the item descriptions.
level_2: String, the desired level 2 category to be found within the list of uncategorized items.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
df: DataFrame, the uncategorized data ready for use with the algorithm.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def uncat_data_prep(inputcsv, supplier_delimiter=None):
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

    # Drop nulls and duplicates, then reset the index.
    df = df.dropna(how='any', axis=0)
    df = df.drop_duplicates(subset=['Primary Key'], keep='first')
    df = df.reset_index(drop=True)

    print("The total number of unique, uncategorized items is:" + str(len(df['Primary Key'])))

    return(df)
