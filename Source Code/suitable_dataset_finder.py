from sklearn.model_selection import ParameterGrid
from copy import deepcopy



# This function receives as parameters:
# - c_f that is the percentage of features that can delete per loop repetition
# - c_s that is the percentage of subjects that can delete per loop repetition
# - dataset that is the original dataset
# - max_nan_allowed that indicates the maximus percentage of nans allowed per feature

# Return a list containing:
# [0] - final number of selected features
# [1] - final number of subjects features 
# [2] - selected features with nan values
# and the shrinked dataset

def clean_dataset (c_f, c_s, balance_factor, thr, main_dataset, side_dataset):
    phenotypic = deepcopy(main_dataset)
    clinical = deepcopy(side_dataset)
    max_nan_allowed = 0.3 #definided minimum amount of allowed missing values

    num_subjects, num_features = phenotypic.shape[0], phenotypic.shape[1]

    min_subjects = 0.25 * len(phenotypic)
    
    feature_nan_values = phenotypic.isna().sum()
    max_feature_nan_actual = feature_nan_values.max()
    perc_max_feature_nan = max_feature_nan_actual/num_subjects

    maintain = ['ADOS_TOTAL', 'ADI_R_VERBAL_TOTAL_BV', 'FIQ']

    while perc_max_feature_nan > max_nan_allowed and len(phenotypic)> min_subjects:
        
        feature_nan_values = feature_nan_values.drop(maintain)
        feature_nan_values_sorted = feature_nan_values.sort_values(ascending=True)
        to_drop = round(c_f * num_features)
        if to_drop < 1:
            to_drop = 1
        features_to_drop = feature_nan_values_sorted[num_features-to_drop:].index
        phenotypic.drop(columns=features_to_drop, inplace=True)
            
        num_subjects, num_features = phenotypic.shape[0], phenotypic.shape[1]
        feature_nan_values = phenotypic.isna().sum()
        max_feature_nan_actual = feature_nan_values.max()
        perc_max_feature_nan = max_feature_nan_actual/num_subjects

        if perc_max_feature_nan > max_nan_allowed and len(phenotypic)> min_subjects:

            subject_nan_values = phenotypic.T.isna().sum()
            subject_nan_values_sorted = subject_nan_values.sort_values(ascending=False)        
            to_drop = round(c_s * num_subjects)
            if to_drop < 1:
                to_drop = 1
    
            class_1_index = clinical[clinical['DX_GROUP'] == 1].index
            class_2_index = clinical[clinical['DX_GROUP'] == 2].index

        
            if (len(clinical[clinical['DX_GROUP'] == 1])/num_subjects) >= thr:
                to_drop_class_2 = round(balance_factor * to_drop)
                to_drop_class_1 = to_drop - to_drop_class_2
            else:
                to_drop_class_1 = round(balance_factor * to_drop)
                to_drop_class_2 = to_drop - to_drop_class_1

    
            # Select subjects to drop from each class
            subjects_to_drop_class_1 = [idx for idx in subject_nan_values_sorted.index if idx in class_1_index][:to_drop_class_1]
            subjects_to_drop_class_2 = [idx for idx in subject_nan_values_sorted.index if idx in class_2_index][:to_drop_class_2]


            # Combine subjects to drop
            subjects_to_drop = subjects_to_drop_class_1 + subjects_to_drop_class_2
            
            phenotypic.drop(subjects_to_drop, inplace=True)
            clinical.drop(subjects_to_drop, inplace=True)

            num_subjects, num_features = phenotypic.shape[0], phenotypic.shape[1]
            feature_nan_values = phenotypic.isna().sum()
            max_feature_nan_actual = feature_nan_values.max()
            perc_max_feature_nan = max_feature_nan_actual/num_subjects

    diagnostic_1 = len(clinical[clinical['DX_GROUP'] == 1])
    diagnostic_2 = len(clinical[clinical['DX_GROUP'] == 2])
    balance_perc = diagnostic_1/ num_subjects
    num_subjects, num_features = phenotypic.shape[0], phenotypic.shape[1]
    return [[c_f, c_s, balance_factor, thr], [num_features, num_subjects], [diagnostic_1, diagnostic_2, balance_perc], feature_nan_values], phenotypic, clinical

def suitable_dataset_finder(main_dataset, side_dataset):
    
    param_grid = {
        'c_f': [0.15,  0.2, 0.25, 0.3, 0.35, 0.4],
        'c_s': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'bal': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'thr': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    }


    '''To optimize the function in order to find the best solution we need to search for different values
    of the parameters c_f and c_s. In order to do that, we will perform a grid search. 
    In this way we try 6 values for c_f, 6 values for c_s and all of the combination between them.

    In order to save space, we decide to store only the results that ensure to conserve at least
    1/5 of the total amount of subjects (or 200) and that gives a variety of at least more than 6 features.'''


    param_combinations = ParameterGrid(param_grid)

    stored_outcomes = []

    rows = main_dataset.shape[0]
    for params in param_combinations:

        c_f = params.get('c_f')
        c_s = params.get('c_s')
        bal = params.get('bal')
        thr = params.get('thr')

        outcome, main_data_cleaned, side_data_cleaned = clean_dataset (c_f, c_s, bal, thr, main_dataset, side_dataset)
        

        condition_1 = (outcome[1][1] >= 200) #num subjects
        condition_2 = (outcome[1][0] > 6)

        num_subjects = main_data_cleaned.shape[0]
        feature_nan_values = main_data_cleaned.isna().sum()
        max_feature_nan_actual = feature_nan_values.max()
        perc_max_feature_nan = max_feature_nan_actual/num_subjects

        condition_3 = perc_max_feature_nan <= 0.30

        if condition_3 and condition_2 and condition_1:
            stored_outcomes.append(outcome)

    if not stored_outcomes:
        return "No suitable datasets found."

    best_results = sorted(stored_outcomes, key=lambda x: x[2][2])

    for result in best_results[0:10]:  # print the best 10 results
        print("With balance: " + str(result[2][2]))
        print(result)

    return best_results