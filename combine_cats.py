'''
Author: Logan Emery
Last updated: 11-28-2018
Organization: ROi
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
