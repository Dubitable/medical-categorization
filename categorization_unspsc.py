def getdf_one(data, level_2):
    import pandas as pd

    df = pd.read_csv(data, encoding='utf-8')

    catdf = df[df['Current ROi Contract Category Level 2'].notnull()]
    catdf = catdf[catdf['ROi Supplier Name'].notnull()]
    catdf = catdf[catdf['ROi Supplier Item Number'].notnull()]

    uncatdf = df[df['Current ROi Contract Category Level 2'].isnull()]
    uncatdf = uncatdf[uncatdf['ROi Supplier Name'].notnull()]
    uncatdf = uncatdf[uncatdf['ROi Supplier Item Number'].notnull()]

    catsupplierseries = catdf['ROi Supplier Name'].str.replace(" ", "_")
    uncatsupplierseries = uncatdf['ROi Supplier Name'].str.replace(" ", "_")

    catdf['Modified ROi Item Description'] = catsupplierseries + " " + catdf['ROi Item Description']
    uncatdf['Modified ROi Item Description'] = uncatsupplierseries + " " + uncatdf['ROi Item Description']

    catdf['Modified ROi Supplier Item Number'] = catsupplierseries + " " + catdf['ROi Supplier Item Number']
    uncatdf['Modified ROi Supplier Item Number'] = uncatsupplierseries + " " + uncatdf['ROi Supplier Item Number']

    catdf = catdf.drop_duplicates(subset=['Modified ROi Supplier Item Number'], keep='first')
    uncatdf = uncatdf.drop_duplicates(subset=['Modified ROi Supplier Item Number'], keep='first')

    catdf = catdf.reset_index()
    uncatdf = uncatdf.reset_index()

    catseries = catdf['Current ROi Contract Category Level 2']
    catlist = catseries.tolist()
    numlist = []

    for i in catlist:
        if i == level_2:
            numlist.append(1)
        else:
            numlist.append(0)

    numseries = pd.Series(numlist).rename('Level 2 Category Key')

    newcatdf = pd.concat([catdf['Modified ROi Supplier Item Number'], catdf['Modified ROi Item Description'], catdf['ROi Supplier Item Number'], catdf['ROi Supplier Name'], catdf['ROi Item Description'], numseries, catdf['Current ROi Contract Category Level 2']], axis=1)
    newuncatdf = pd.concat([uncatdf['Modified ROi Supplier Item Number'], uncatdf['Modified ROi Item Description'], uncatdf['ROi Supplier Item Number'], uncatdf['ROi Supplier Name'], uncatdf['ROi Item Description']], axis=1)

    return(newcatdf, newuncatdf)


def fixfinaldf(uncatdf, y_test, catdict):
    import pandas as pd
    catdescriptions = []

    for key in y_test:
        catdescriptions.append(catdict[key])

    uncatdf['Level 2 Key'] = y_test
    uncatdf['Level 2 Description'] = catdescriptions

    return(uncatdf)


def combine_cats(inputcsv):
    import pandas as pd

    df = pd.read_csv(inputcsv, encoding='latin1')

    # Setting up the dataframe by providing a primary key and including the supplier name in the description.
    primary_key = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Supplier Item Number']).rename('Primary Key')
    modified_description = (df['ROi Supplier Name'] + " " + df['ROi Item Description']).rename('Modified Description')
    df = pd.concat([primary_key, modified_description, df['ROi Supplier Name'], df['ROi Supplier Item Number'], df['ROi Item Description'], df['Current ROi Contract Category Level 2']], axis=1)

    df = df.dropna(how='any', axis=0)
    df = df.drop_duplicates(subset=['Primary Key'], keep='first')
    df = df.reset_index(drop=True)

    working = df['Current ROi Contract Category Level 2']

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

    for key in replace_dict:
        working = working.replace(key, replace_dict[key])

    combined_cats = working.rename('Modified Category')

    df = pd.concat([df, combined_cats], axis=1)

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

    final_df = pd.concat([df, cat_num_series], axis=1)

    return(final_df, cat_dict)


def get_catdict(df):

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

    df = df.reset_index().iloc[:, 1:]

    finaldf = pd.concat([df, cat_num_series], axis=1)

    print("The total number of unique categories is:" + str(len(cat_dict)))

    return(finaldf, cat_dict)


def unspsc_data_prep(inputcsv, supplier_delimiter='space'):
    import pandas as pd

    df = pd.read_csv(inputcsv, encoding='latin1')
    df = df[df['UNSPSC 4 Top 90% Spend'] == 'Top 90% Spend']

    # Setting up the dataframe by providing a primary key and including the supplier name in the description.
    if supplier_delimiter == 'underscore':
        primary_key = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Supplier Item Number']).rename('Primary Key')
        modified_description = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Item Description']).rename('Modified Description')
    elif supplier_delimiter == 'space':
        primary_key = (df['ROi Supplier Name'] + " " + df['ROi Supplier Item Number']).rename('Primary Key')
        modified_description = (df['ROi Supplier Name'] + " " + df['ROi Item Description']).rename('Modified Description')
    elif supplier_delimiter == None:
        primary_key = (df['ROi Supplier Name'] + " " + df['ROi Supplier Item Number']).rename('Primary Key')
        modified_description = df['ROi Item Description'].rename('Modified Description')

    df = pd.concat([primary_key, modified_description, df['ROi Supplier Name'], df['ROi Supplier Item Number'], df['ROi Item Description'], df['Item UNSPSC Code']], axis=1)

    df = df.dropna(how='any', axis=0)
    df = df.drop_duplicates(subset=['Primary Key'], keep='first')

    df['UNSPSC Code 1'] = df['Item UNSPSC Code'].astype(str).str[:2]
    df['UNSPSC Code 2'] = df['Item UNSPSC Code'].astype(str).str[2:4]
    df['UNSPSC Code 3'] = df['Item UNSPSC Code'].astype(str).str[4:6]
    df['UNSPSC Code 4'] = df['Item UNSPSC Code'].astype(str).str[6:8]


    # FIRST CODE DICT
    cat_series = df['UNSPSC Code 1']

    cat_list = cat_series.tolist()
    cat_list_unique = list(set(cat_list))

    cat_keys = list(range(len(cat_list_unique)))
    cat_dict1 = dict.fromkeys(cat_keys)
    t = 0
    while t < len(cat_dict1):
        cat_dict1[t] = cat_list_unique[t]
        t += 1

    inverted_catdict = {v: k for k, v in cat_dict1.items()}

    cat_num_list = []
    for cat in cat_list:
        cat_num_list.append(inverted_catdict[cat])

    cat_num_series1 = pd.Series(cat_num_list).rename('UNSPSC Key 1')
    

    # SECOND CODE DICT
    cat_series = df['UNSPSC Code 2']

    cat_list = cat_series.tolist()
    cat_list_unique = list(set(cat_list))

    cat_keys = list(range(len(cat_list_unique)))
    cat_dict2 = dict.fromkeys(cat_keys)
    t = 0
    while t < len(cat_dict2):
        cat_dict2[t] = cat_list_unique[t]
        t += 1

    inverted_catdict = {v: k for k, v in cat_dict2.items()}

    cat_num_list = []
    for cat in cat_list:
        cat_num_list.append(inverted_catdict[cat])

    cat_num_series2 = pd.Series(cat_num_list).rename('UNSPSC Key 2')


    # THIRD CODE DICT
    cat_series = df['UNSPSC Code 3']

    cat_list = cat_series.tolist()
    cat_list_unique = list(set(cat_list))

    cat_keys = list(range(len(cat_list_unique)))
    cat_dict3 = dict.fromkeys(cat_keys)
    t = 0
    while t < len(cat_dict3):
        cat_dict3[t] = cat_list_unique[t]
        t += 1

    inverted_catdict = {v: k for k, v in cat_dict3.items()}

    cat_num_list = []
    for cat in cat_list:
        cat_num_list.append(inverted_catdict[cat])

    cat_num_series3 = pd.Series(cat_num_list).rename('UNSPSC Key 3')


    # FOURTH CODE DICT
    cat_series = df['UNSPSC Code 4']

    cat_list = cat_series.tolist()
    cat_list_unique = list(set(cat_list))

    cat_keys = list(range(len(cat_list_unique)))
    cat_dict4 = dict.fromkeys(cat_keys)
    t = 0
    while t < len(cat_dict4):
        cat_dict4[t] = cat_list_unique[t]
        t += 1

    inverted_catdict = {v: k for k, v in cat_dict4.items()}

    cat_num_list = []
    for cat in cat_list:
        cat_num_list.append(inverted_catdict[cat])

    cat_num_series4 = pd.Series(cat_num_list).rename('UNSPSC Key 4')


    df = df.reset_index(drop=True)

    finaldf = pd.concat([df, cat_num_series1, cat_num_series2, cat_num_series3, cat_num_series4], axis=1)

    print("The total number of unique, categorized items is:" + str(len(df['Primary Key'])))
    print("The total number of unique UNSPSC 1's is:" + str(len(cat_dict1)))
    print("The total number of unique UNSPSC 2's is:" + str(len(cat_dict2)))
    print("The total number of unique UNSPSC 3's is:" + str(len(cat_dict3)))
    print("The total number of unique UNSPSC 4's is:" + str(len(cat_dict4)))

    return(finaldf, cat_dict1, cat_dict2, cat_dict3, cat_dict4)


def uncat_data_prep(inputcsv):
    import pandas as pd

    df = pd.read_csv(inputcsv, encoding='latin1')

    primary_key = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Supplier Item Number']).rename('Primary Key')
    modified_description = (df['ROi Supplier Name'].str.replace(" ", "_") + " " + df['ROi Item Description']).rename('Modified Description')
    df = pd.concat([primary_key, modified_description, df['ROi Supplier Name'], df['ROi Supplier Item Number'], df['ROi Item Description']], axis=1)

    df = df.dropna(how='any', axis=0)
    df = df.drop_duplicates(subset=['Primary Key'], keep='first')

    df = df.reset_index(drop=True)

    print("The total number of unique, uncategorized items is:" + str(len(df['Primary Key'])))

    return(df)


def pre_process(catdf, uncatdf=False, supplier_count_threshold=1, int_thresh=False, int_std_scale=1):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    #   1) Remove any supplier with less than x items in total.
    supplier_counts = catdf['ROi Supplier Name'].value_counts()
    supplier_fail = supplier_counts[supplier_counts <= supplier_count_threshold].index.tolist()
    msk = catdf['ROi Supplier Name'].isin(supplier_fail)
    supplier_fail_df = catdf[msk].reset_index(drop=True)
    catdf = catdf[~msk].reset_index(drop=True)
    print('Number of suppliers removed due to having too few items in training set:' + str(len(supplier_fail)))
    print('Number of items removed due to supplier having too few items in training set:' + str(len(supplier_fail_df)))

    #   2) Generate a threshold for the amount of information contained by an item description.
    cat_desc = catdf['Modified Description'].as_matrix()

    cat_tfidf = TfidfVectorizer()
    cat_tfidf.fit(cat_desc)

    cat_idfs = cat_tfidf.idf_
    cat_vocab = cat_tfidf.vocabulary_
    cat_idf_dict = dict(zip(cat_vocab, cat_idfs))
    cat_documents = cat_tfidf.inverse_transform(cat_tfidf.transform(cat_desc))

    cat_ivs = []

    for document in cat_documents:
        cat_iv = 0
        for term in document:
            cat_iv += cat_idf_dict[term]
        cat_ivs.append(cat_iv)

    integrity_mean = np.mean(cat_ivs)
    integrity_std = np.std(cat_ivs)

    if int_thresh != False:
        integrity_thresh = int_thresh
    else:
        integrity_thresh = integrity_mean - (int_std_scale * integrity_std)

    cat_ivs_series = pd.Series(cat_ivs).rename('Integrity Value')
    catdf = pd.concat([catdf, cat_ivs_series], axis=1)
    cat_pass_df = catdf[catdf['Integrity Value'] > integrity_thresh].reset_index(drop=True)
    cat_fail_df = catdf[catdf['Integrity Value'] <= integrity_thresh].reset_index(drop=True)

    print('Number of items removed due to low integrity item descriptions:' + str(len(cat_fail_df)))

    # Note that the uncatdf is having its integrity values compared to the threshold set by the catdf (above)
    # if type(uncatdf) == 'pandas.core.frame.DataFrame': #boolian needs work ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     desc_series = pd.concat([df['Modified Description'], uncatdf['Modified Description']], axis=0)
    #     desc_array = desc_series.as_matrix()
    #
    #     uncat_series = uncatdf['Modified Description']
    #     uncat_array = uncat_series.as_matrix()
    #
    #     tfidf = TfidfVectorizer()
    #     tfidf.fit(desc_array)
    #
    #     idfs = tfidf.idf_
    #     vocab = tfidf.vocabulary_
    #     idf_dict = dict(zip(vocab, idfs))
    #     documents = tfidf.inverse_transform(tfidf.transform(uncat_array))
    #
    #     uncat_integrity_values = []
    #
    #     for document in documents:
    #         uncat_integrity_value = 0
    #         for term in document:
    #             uncat_integrity_value += idf_dict[term]
    #         uncat_integrity_values.append(uncat_integrity_value)
    #
    #     uncat_integrity_values_df = pd.Series(uncat_integrity_values).rename('Integrity Value')
    #     uncatdf = pd.concat([uncatdf, uncat_integrity_values_df], axis=1)
    #     uncat_pass_df = uncat_df[uncat_df['Integrity Value'] > integrity_thresh].reset_index(drop=True)
    #     uncat_fail_df = uncat_df[uncat_df['Integrity Value'] <= integrity_thresh].reset_index(drop=True)
    #
    #     print("The number of items dropped from the uncategorized set is:" + str(len(uncat_fail_df)))
    #
    # else:
    #     uncat_pass_df = 0
    #     uncat_fail_df = 0

    return(cat_pass_df, cat_fail_df, integrity_thresh)


def infodensity_test(df, normalized=False, id_thresh=False):
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    desc_series = df['Modified Description']
    desc_array = desc_series.as_matrix()

    tfidf = TfidfVectorizer()
    tfidf.fit(desc_array)

    idfs = tfidf.idf_
    vocab = tfidf.vocabulary_
    idf_dict = dict(zip(vocab, idfs))
    documents = tfidf.inverse_transform(tfidf.transform(desc_array))

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

    id_mean = np.mean(id_values)
    id_stdev = np.std(id_values)

    if id_thresh != False:
        info_thresh = id_thresh
    else:
        info_thresh = id_mean - id_stdev

    id_df = pd.Series(id_values).rename('Total IDF')

    doc_lens_df = pd.Series(document_lens).rename('Number of Terms')

    df = pd.concat([df, id_df, doc_lens_df], axis=1)
    #pass_df = df[df['Information Density Value'] > integrity_thresh].reset_index(drop=True)
    #fail_df = df[df['Information Density Value'] <= integrity_thresh].reset_index(drop=True)

    return(df, info_thresh)


def cat_accuracy(datacsv, train_size):
    import pandas as pd
    import numpy as np
    from categorization import cat_data_prep
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    df, catdict = cat_data_prep(datacsv)

    catsize_dict = dict.fromkeys(catdict)
    for key in catsize_dict:
        catsize_dict[key] = len(df[df['Category Key'] == key])

    # Generate training and testing sets using numpy's ability to take a random sample in [0,1)
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
    y_output = text_clf.predict(X_test)

    y_output_series = pd.Series(y_output).rename('Proposed Category Key')

    proposed_cats = []
    for key in y_output:
        proposed_cats.append(catdict[key])

    proposed_cats_series = pd.Series(proposed_cats).rename('Proposed Category Description')

    test_df2 = pd.concat([test_df, y_output_series, proposed_cats_series], axis=1)

    # Create a new df with the accuracies of each Category
    unique_catnum_list = list(catdict.keys())
    unique_cat_list = []
    for i in unique_catnum_list:
        unique_cat_list.append(catdict[i])

    unique_catnum_series = pd.Series(unique_catnum_list).rename('Level 2 Key')
    unique_cat_series = pd.Series(unique_cat_list).rename('Level 2 Description')

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

    accuracy_score_series = pd.Series(accuracy_scores).rename('Algorithm Accuracy')
    testing_size_series = pd.Series(testing_size).rename('Testing Set Size')
    category_size_series = pd.Series(category_size).rename('Total Category Size')

    accuracy_df = pd.concat([unique_catnum_series, unique_cat_series, accuracy_score_series, testing_size_series, category_size_series], axis=1)

    return(accuracy_df)
