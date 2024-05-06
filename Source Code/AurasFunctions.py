
'''
We decide that to fill the missing values of the test subministred
it should be good to rely on the standard score achieved by the mean
of the population (if the statistics are available in the literature),
otherwise we will use the mean extracted from our dataset.
 '''

def test_score_fill (feature_name, feature_data):
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
    

# Lista delle coppie di features da controllare
feature_pairs = [
    ('FIQ_TEST_TYPE', 'FIQ'),
    ('PIQ_TEST_TYPE', 'PIQ'),
    ('VIQ_TEST_TYPE', 'VIQ')]

# Iteriamo su ogni coppia di features
for test_type_col, score_col in feature_pairs:
    # Iteriamo su ogni riga del DataFrame
    for index, row in ASD_phenotypic_filtered.iterrows():
        # Controlliamo se il valore nella colonna 'test_type_col' è mancante
        if pd.isnull(row[test_type_col]):
            # Se il valore nella colonna 'score_col' è presente
            if not pd.isnull(row[score_col]):
                # Calcoliamo la moda di 'test_type_col'
                mode_test_type = ASD_phenotypic_filtered[test_type_col].mode()[0]
                # Sostituiamo il valore mancante nella colonna 'test_type_col' con la moda
                ASD_phenotypic_filtered.at[index, test_type_col] = mode_test_type