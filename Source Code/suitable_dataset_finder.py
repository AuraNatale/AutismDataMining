from sklearn.model_selection import ParameterGrid



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

def clean_dataset (c_f, c_s, balance_factor, main_dataset, side_dataset):
    ASD_phenotypic = main_dataset.copy()
    ASD_clinical = side_dataset.copy()
    max_nan_allowed = 0.1 #definided minimum amount of allowed missing values

    num_subjects, num_features = ASD_phenotypic.shape[0],ASD_phenotypic.shape[1]

    min_subjects = 0.25 * len(main_dataset)
    
    feature_nan_values = ASD_phenotypic.isna().sum()
    max_feature_nan_actual = feature_nan_values.max()
    perc_max_feature_nan = max_feature_nan_actual/num_subjects

    maintain = ['ADOS_TOTAL', 'ADI_R_VERBAL_TOTAL_BV', 'FIQ']

    while perc_max_feature_nan > max_nan_allowed and len(ASD_phenotypic)> min_subjects:
        
        filtered_feature_nan_values_sorted = feature_nan_values.drop(maintain)
        feature_nan_values_sorted = filtered_feature_nan_values_sorted.sort_values(ascending=True)
        to_drop = round(c_f * num_features)
        features_to_drop = feature_nan_values_sorted[num_features-to_drop:].index
        ASD_phenotypic.drop(columns=features_to_drop, inplace=True)
            
        num_subjects, num_features = ASD_phenotypic.shape[0],ASD_phenotypic.shape[1]

        subject_nan_values = ASD_phenotypic.T.isna().sum()
        subject_nan_values_sorted = subject_nan_values.sort_values(ascending=False)        
        to_drop = round(c_s * num_subjects)
   
        class_1_index = ASD_clinical[ASD_clinical['DX_GROUP'] == 1].index
        class_2_index = ASD_clinical[ASD_clinical['DX_GROUP'] == 2].index

     
        if (len(ASD_clinical[ASD_clinical['DX_GROUP'] == 1])/num_subjects) >= 0.65:
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
        
        ASD_phenotypic.drop(subjects_to_drop, inplace=True)
        ASD_clinical.drop(subjects_to_drop, inplace=True)

        feature_nan_values = ASD_phenotypic.isna().sum()
        max_feature_nan_actual = feature_nan_values.max()
        perc_max_feature_nan = max_feature_nan_actual/num_subjects

    num_subjects, num_features = ASD_phenotypic.shape[0],ASD_phenotypic.shape[1]
    return [[c_f, c_s, balance_factor], [num_features, num_subjects], feature_nan_values], ASD_phenotypic, ASD_clinical


param_grid = {
    'c_f': [0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075,
             0.08, 0.085, 0.9, 0.095, 0.1, 0.105, 0.11, 0.115, 0.2, 0.25, 0.3],
    'c_s': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
             0.075,  0.08, 0.085, 0.9, 0.095, 0.1,0.105, 0.11, 0.115, 0.2, 0.25, 0.3],
    'bal': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
}


'''To optimize the function in order to find the best solution we need to search for different values
 of the parameters c_f and c_s. In order to do that, we will perform a grid search. 
 In this way we try 22 values for c_f and 24 values for c_s and all of the combination between them.

In order to save space, we decide to store only the results that ensure to conserve at least
 1/4 of the total amount of subjects and that gives a variety of at least more than 10 features.'''


param_combinations = ParameterGrid(param_grid)

stored_outcomes = []

rows = ASD_phenotypic.shape[0]
for params in param_combinations:

    c_f = params.get('c_f')
    c_s = params.get('c_s')
    bal = params.get('bal')

    outcome, main_data_cleaned, side_data_cleaned = clean_dataset (c_f, c_s, bal, ASD_phenotypic, ASD_clinical)
    

    condition_1 = (outcome[1][1] >= 0.2 * rows)
    condition_2 = (outcome[1][0] > 8)

    num_subjects = main_data_cleaned.shape[0]
    feature_nan_values = main_data_cleaned.isna().sum()
    max_feature_nan_actual = feature_nan_values.max()
    perc_max_feature_nan = max_feature_nan_actual/num_subjects

    print(perc_max_feature_nan)
    condition_3 = perc_max_feature_nan <= 0.30

    if condition_3 and condition_2 and condition_1:
        print("store!")
        stored_outcomes.append(outcome)

'''In order to choose one of them, we decide to maintain the one that maintain
 better differenciation between the two classes of the diagnostic.'''

#Find ten best results

best_results = [[1000,0,0,0], [1000,0,0,0], [10000,0,0,0], [1000,0,0,0], [1000,0,0,0], [10000,0,0,0], [1000,0,0,0],
                 [1000,0,0,0], [10000,0,0,0], [10000,0,0,0]]

for i in range (len(stored_outcomes)):
    class_counts = stored_outcomes[i][1]
    balance = class_counts[0]-class_counts[1]
    differences_score = []
    for j in range (len(best_results)):
        difference = balance - best_results[j][0]
        differences_score.append(difference)
    min_value = min(differences_score)
    if min_value < 0:
        index_max = differences_score.index(min_value)
        best_results[index_max] = [class_counts[0]/stored_outcomes[i][1][1], stored_outcomes[i][0], stored_outcomes[i][1], stored_outcomes[i][2]]

sorted_best_results = sorted(best_results, key=lambda x: x[0])
for result in sorted_best_results:

    print(result)