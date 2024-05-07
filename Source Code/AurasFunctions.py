import pandas as pd

'''
We decide that to fill the missing values of the test subministred
it should be good to rely on the standard score achieved by the mean
of the population (if the statistics are available in the literature),
otherwise we will use the mean extracted from our dataset.
 '''

#How to use the function:
#result = df['A'].apply(custom_function, args=(multiplier,))
# dataset[feature_name] = dataset[feature_name].apply(test_score_fill, args=(feature_name, feature_mean))
# dataset[feature_name] = dataset[feature_name].apply(lambda x: test_score_fill(feature_name, x))
def test_score_fill (feature_value, feature_name, feature_mean):
    # We create a dictionary to store the literature mean scores
    literature_scores = {
    "FIQ": 97.34, #retrieved from 
    "VIQ": 97.34, #retrieved from 
    "PIQ": 97.34, #retrieved from 
    "ADOS_COMM": "not available", #retrieved from 
    "ADOS_SOCIAL": "not available", #retrieved from 
    "ADI_R_SOCIAL_TOTAL_A": "not available", #retrieved from 
    "ADI_R_VERBAL_TOTAL_BV": "not available", #retrieved from 
    "ADI_RRB_TOTAL_C": "not available", #retrieved from 
    "SRS_RAW_TOTAL": "not available", #retrieved from 
    }

    # Then we check which feature we obtained to decide if replace
    # using the value in the dictionary ot directly the mean of the data
    if pd.isna(feature_value):

        if feature_name in literature_scores:
            return literature_scores[feature_name]
        else:
            return feature_mean
    else:

        return feature_value
    
   