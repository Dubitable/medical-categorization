# medical-categorization
A collection of scripts utilizing a machine learning algorithm to categorize new medical supplies based on short-text descriptions of previously categorized items.


The scripts are sorted into three distinct categories:

- Data Prep and Algorithm: all scripts used to prepare the data for the model and to run the model itself.
- Helper Functions: scripts that can be used to improve data quality.  Includes cleaning of the categories themselves (specific to the dataset), and pre-processing.
- Exploratory Functions: functions that can be used to generate metrics to explore the data.  Specifically, the integrity of the data and the accuracy of each category.


All scripts are heavily commented with their intended function, arguments, and outputs.  An order of the indended use of the scripts is outlined below.


Order:

1. combine_cats (optional)
2. cat_data_prep
3. pre_process (optional)
4. uncat_data_prep
5. run_svm
6. infodensity_test (optional)
7. cat_accuracy (optional)
8. fixfinaldf
