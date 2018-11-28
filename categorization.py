'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
'''



'''
------------------------------------------------ DATA PREP AND ALGORITHM ------------------------------------------------
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



'''
------------------------------------------------ HELPER FUNCTIONS ------------------------------------------------
'''

'''
~~~~~~~~ combine_cats ~~~~~~~~
Takes the original list of categories and combines those which should be the same category.
Mostly accounts for spelling/punctuation errors.
MUST BE MANUALLY MAINTAINED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arguments:
inputcsv: .csv filename as string, the raw data file containing all categorized item data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Returns:
final_df: DataFrame, the categorized data containing the fixed categories.
cat_dict: Dictionary, the fixed categories and their keys.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def combine_cats(inputcsv):
    import pandas as pd

    df = pd.read_csv(inputcsv, encoding='latin1')

    # Setting up the dataframe by providing a primary key and including the supplier name in the description.
    primary_key = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Supplier Item Number']).rename('Primary Key')
    modified_description = (df['ROi Supplier Name'] + " " + df['ROi Item Description']).rename('Modified Description')
    df = pd.concat([primary_key, modified_description, df['ROi Supplier Name'], df['ROi Supplier Item Number'], df['ROi Item Description'], df['Current ROi Contract Category Level 2']], axis=1)

    # Drop duplicates and nulls.
    df = df.dropna(how='any', axis=0)
    df = df.drop_duplicates(subset=['Primary Key'], keep='first')
    df = df.reset_index(drop=True)

    # The complete dictionary of erroneous categories, and their replacements.  This part must be maintained.
    replace_dict = {'Ablation, Radiofrequency ; Ultrasound Equipment': 'Ablation, Radiofrequency; Ultrasound Equipment',
                    'Anesthesia Commodity Products ; Spinal Trays and Supplies': 'Anesthesia Commodity Products; Spinal Trays and Supplies',
                    'Automated Microbial Identification and Susceptibility Analyzeres, Reagents and Consumables': 'Automated Microbial Identification and Susceptibility Analyzers, Reagents and Consumables',
                    'Bags, Plastic Can Liners': 'Bags, Plastic',
                    'Blood Pressure Cuffs & Supplies; Patient Monitoring, Equipment and Accessories': 'Blood Pressure Cuffs & Supplies',
                    'Bone & Biologics': 'Bone and Biologics',
                    'BONE CEMENT AND MIXING PRODUCTS': 'Bone Cement and Mixing Products',
                    'Bowel and Fecal Management ': 'Bowel and Fecal Management',
                    'branded': 'Branded',
                    'Can Liners': 'Bags, Plastic',
                    'Cardiac Surgery, Products': 'Cardiac Surgery Products',
                    'Catheters, Hemodialysis ; Catheters, Venous and Arterial': 'Catheters, Hemodialysis; Catheters, Venous and Arterial',
                    'CLINICAL': 'Clinical Commodity Products',
                    'Clinical': 'Clinical Commodity Products',
                    'Computer, Hardware ; Master Service Agreement': 'Computer, Hardware',
                    'Constrast Media': 'Contrast Media',
                    'Contrast Media ': 'Contrast Media',
                    'Devices, IV Securement ': 'Devices, IV Securement',
                    'Diagnostic Imaging Capital Equipment': 'Diagnostic Imaging - Capital, Radiology, Film',
                    'Disinfectants, Patient, Alcohol Prep Pads': 'Disinfectants, Patient, Alcohol Prep Pads and Swabs',
                    'Electrosurgical, Generators and Disposables ; Ultrasonic Devices, Electrosurgical Generators & Disposables': 'Electrosurgical, Generators and Disposables',
                    'Electrosurgical, Generators and Disposables; Ultrasonic Devices, Electrosurgical Generators & Disposables': 'Electrosurgical, Generators and Disposables',
                    'Endoscopy Equipment and Accessories': 'Endoscopy, Equipment and Accessories',
                    'endo-Suture': 'Endo-Suture',
                    'EQUIPMENT, RENTAL': 'Equipment, Rental',
                    'IMPLANTS, TOTAL JOINTS': 'Implants, Total Joints',
                    'implants, trauma': 'Implants, Trauma',
                    'IMPLANTS, TRAUMA': 'Implants, Trauma',
                    'Intra-Aortic Balloon Pumps (IABP) and Equipment': 'Intra-Aortic Balloon Pumps (IABP) & Equipment',
                    'IV Bags, Infuser': 'IV Bags and Infusers',
                    'Medical Bandages ': 'Medical Bandages',
                    'OBGYN, Miscellaneous Disposables': 'OB/GYN, Miscellaneous Disposables',
                    'Obstetric commodity products': 'Obstetric Commodity Products',
                    'Office Supplies - General / Paper / Printing / Copy': 'Office Supply',
                    'patient cleansing': 'Patient Cleansing',
                    'PATIENT CLEANSING': 'Patient Cleansing',
                    'Patient Handling, Equipment & Medical Equipment': 'Patient Handling, Equipment',
                    'Patient Monitoring Electrodes and Supplies': 'Patient Monitoring, Electrodes & Supplies',
                    'Patient Safety Equipment': 'Patient Safety Products',
                    'Patient Slippers  ': 'Patient Slippers',
                    'Personal Protective, Equipment ; Uniforms & Apparel  ; Masks, Surgical': 'Personal Protective, Equipment; Uniforms & Apparel; Masks, Surgical',
                    'Preoperative Skin Prep & Patient Cleansing': 'Preoperative Skin Prep',
                    'Prep, Surgical Skin ': 'Prep, Surgical Skin',
                    'Reference Lab': 'Reference Lab Services',
                    'Regulated Medical Waste': 'Regulated Medical Waste Services, Sharps Management Services, Hazardous Waste Services and Pharmacy Waste Services',
                    'respiratory Therapy Supplies and Equipment': 'Respiratory Therapy Supplies and Equipment',
                    'RFID Tags': 'RFID',
                    'SAFETY': 'Safety',
                    'Sanitation/Janitorial Services and Supplies': 'Sanitation/Janitorial Services & Supplies',
                    'Solutions, Disposables and Equipment': 'Solutions, Disposables & Equipment',
                    'SURGICAL POWER EQUIPMENT AND RELATED DISPOSABLES': 'Surgical Power Equipment and Related Disposables',
                    'Ultrasonic Devices, Electrosurgical Generators and Disposables': 'Ultrasonic Devices, Electrosurgical Generators & Disposables',
                    'Uniforms & Apparel  ; Floor Mats and Cleaning Service': 'Uniforms & Apparel; Floor Mats and Cleaning Service',
                    'Uniforms & Apparel  ; Textiles, Sewn Goods, Linens': 'Uniforms & Apparel; Textiles, Sewn Goods, Linens'}

    # Instantiate a working copy of the category column, then replace each erroneous category and add to original dataframe.
    working = df['Current ROi Contract Category Level 2']
    for key in replace_dict:
        working = working.replace(key, replace_dict[key])
    new_cats = working.rename('Modified Category')
    df = pd.concat([df, new_cats], axis=1)

    # Create the (new) category dictionary for reference later.
    catnumlist = []
    cat_series = df['Modified Category']
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

    # Add the (new) category keys to the final dataframe.
    final_df = pd.concat([df, cat_num_series], axis=1)

    return(final_df, cat_dict)



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



'''
------------------------------------------------ EXPLORATORY FUNCTIONS ------------------------------------------------
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
