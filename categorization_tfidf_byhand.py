def gettfidf(inputdata):
    import numpy as np
    import pandas as pd
    import math
    pd.options.mode.chained_assignment = None # default='warn'

# Import the data set as a csv file.
    if inputdata[len(inputdata)-4:] != '.csv':
        inputdata += '.csv'

# Filter the data by nonempty item descriptions
    df = pd.read_csv(inputdata)
    df_filtered = df[df['ROi Item Description'].notnull()]
    df_filtered2 = df_filtered[df_filtered['ROi Supplier Name'].notnull()]
    df = df_filtered2

# Concatenate the Item Description with the Supplier, with a space between
    df['Modified ROi Item Description'] = df['ROi Supplier Name'] + " " + df['ROi Item Description']

# Remove duplicate items based on the modified column and remove undesired characters
    df = df.drop_duplicates(subset=['Modified ROi Item Description'], keep='first')

    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.lower()
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(",", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(";", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(":", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace("-", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace("(", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(")", " ")

# Extract the category and the item description
    tfidf = pd.concat([df['Current ROi Contract Category Level 2'], df['Modified ROi Item Description']], axis=1)

# Generate a list of only the item descriptions, then a list of each individual term that is unique.
    itemlist = tfidf['Modified ROi Item Description'].tolist()
    termlist = []

# Remove useless "" and split the item into terms
    for item in itemlist:
        for term in item.split(" "):
            if term != "":
                termlist.append(term)

    termset = set(termlist)
    termlist = list(termset)

    idf = []

# Count the number of item descriptions that contain each term
    for term in termlist:
        i = 0
        for item in tfidf['Modified ROi Item Description']:
            if item.split().count(term) > 0:
                i += 1 # i should be the number of item descriptions that contain the term in question
        if i != 0:
            idf.append([term, math.log(len(tfidf['Modified ROi Item Description'])/i)])

# Take the term/idf pair, generate a pd series of the tfidf for that term, then tack it onto the tfidf dataframe
    for pair in idf:
        tflist = []
        for item in itemlist:
            tflist.append(item.split().count(pair[0])*pair[1])
        tfseries = pd.Series(tflist)
        tfseries.name = pair[0]
        tfidf[pair[0]] = tfseries

    return(df, tfidf)



def getbagofwords(inputdata):
    import numpy as np
    import pandas as pd
    pd.options.mode.chained_assignment = None # default='warn'

# Import the data set as a csv file.
    if inputdata[len(inputdata)-4:] != '.csv':
        inputdata += '.csv'

# Filter the data by nonempty item descriptions
    df = pd.read_csv(inputdata)
    df_filtered = df[df['ROi Item Description'].notnull()]
    df_filtered2 = df_filtered[df_filtered['ROi Supplier Name'].notnull()]
    df = df_filtered2

# Concatenate the Item Description with the Supplier, with a space between
    df['Modified ROi Item Description'] = df['ROi Supplier Name'] + " " + df['ROi Item Description']

# Remove duplicate items based on the modified column and remove undesired characters
    df = df.drop_duplicates(subset=['Modified ROi Item Description'], keep='first')

    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.lower()
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(",", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(";", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(":", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace("-", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace("(", " ")
    df['Modified ROi Item Description'] = df['Modified ROi Item Description'].str.replace(")", " ")

# Extract the category and the item description
    bagofwords = pd.concat([df['Current ROi Contract Category Level 2'], df['Modified ROi Item Description']], axis=1)

# Generate a list of only the item descriptions, then a list of each individual term that is unique.
    itemlist = bagofwords['Modified ROi Item Description'].tolist()
    termlist = []

# Remove useless "" and split the item into terms
    for item in itemlist:
        for term in item.split(" "):
            if term != '':
                termlist.append(term)

    termset = set(termlist)
    termlist = list(termset)

# Count the number of times a word appears in each item description, and append to the bagofwords df

    for term in termlist:
        bagofwordslist = []
        for item in itemlist:
            bagofwordslist.append(item.count(term))
        bagofwordsseries = pd.Series(bagofwordslist)
        bagofwords[term] = bagofwordsseries

    return(df, bagofwords)



def getnumtfidf(tfidf):
    import numpy as np
    import pandas as pd

    categoriesdf = tfidf.ix[:,0]

# Take the category series and convert to a list, then get a unique set
    categorieslist = categoriesdf.tolist()
    categoriesset = set(categorieslist)
    categorieslistunique = list(categoriesset)

# Turn the set of unique categories into a dictionary of categories with numbered keys
    categorieskeys = list(range(len(categorieslistunique)))
    categoriesdict = dict.fromkeys(categorieskeys)
    i = 0
    while i < len(categoriesdict):
        categoriesdict[i] = categorieslistunique[i]
        i += 1

# Invert the mapping of the dictionary, to help with gettting a list of the numbers
    inverted_categoriesdict = {v: k for k, v in categoriesdict.items()}

# Generate a list, matching the category list, of each of the keys
    categoriesnumlist = []
    for cat in categorieslist:
        categoriesnumlist.append(inverted_categoriesdict[cat])

# Turn number list into dataframe and make new dataframe with numbers
    categoriesnumdf = pd.Series(categoriesnumlist)
    numtfidf = tfidf
    numtfidf['Current ROi Contract Category Level 2'] = categoriesnumdf

    return(numtfidf, categoriesdict)



def getxyarray(numtfidf):
    import numpy as np
    import pandas as pd

# Remove bottom two rows; always seem to be empty
    numtfidf = numtfidf.ix[:len(numtfidf)-1,:]

# Remove all NAN entries
    numtfidf = numtfidf.fillna(0)

# Get series of category codes, and dataframe of corresponding tfidf
    Xdf = numtfidf.ix[:,2:]
    ydf = numtfidf.ix[:,0]

# Convert X and y into numpy arrays
    X = Xdf.as_matrix()
    y = ydf.as_matrix()

    return(X, y)



def getbagxyarray(bagofwords):
    import numpy as np
    import pandas as pd

# Remove bottom two rows; always seem to be empty



def xysplit(X, y, size, state):
    from sklearn.model_selection import train_test_split

# Split the X and y sets into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=state)

    return(X_train, X_test, y_train, y_test)



def gettree(X_train, y_train, state):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import export_graphviz

# Declare a decision tree and fit it to the training set
    model = DecisionTreeClassifier(random_state=state)
    model.fit(X_train, y_train)

    print("Accuracy on test set: {:3f}".format(tree.score(X_test, y_test)))
    filename = input("Name of tree export file?")

# This will create a txt file of your choosing inside the directory that the script is in.
# To visualize, copy and paste into the box in the following link:
# http://webgraphviz.com/
    with open(filename, "w") as f:
        f = export_graphviz(model, out_file=f)
