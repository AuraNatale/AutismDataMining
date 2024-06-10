import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import seaborn as sns
import OurFunctions as of #saperated collection
   
import os #to interact to the file system
import numpy as np #Statistics
import pandas as pd #Database Technology <-> Data preproc & Data Analysis
from matplotlib import pyplot as plt #Visualization
import seaborn as sns #Visualization
import missingno as msno
import random 
from sklearn.preprocessing import RobustScaler #scikit-learn -> ML
import OurFunctions as of #saperated collection


def remove_high_missing(df, key_features, balance_column, min_subjects=200, max_missing_percentage=10):
    current_df = df.copy()
    
    # Calculate starting balancing proportion at first 
    initial_balance = current_df[balance_column].value_counts(normalize=True)
    
    while (of.calculate_missing_percentage(current_df) > max_missing_percentage): 
        # Evaluate the percentahge of missing values for each column and raw
        missing_percent_features = current_df.isna().mean() * 100
        missing_percent_subjects = current_df.isna().mean(axis=1) * 100
        
        # Filter key features to not remove them
        non_key_features = missing_percent_features.drop(labels=key_features, errors='ignore')
        
        # Find the feature or subject with the higher percentage of missing vales
        max_feature_missing = non_key_features.max()
        max_subject_missing = missing_percent_subjects.max()
        
        # Remove it
        if max_feature_missing >= max_subject_missing and not non_key_features.empty:
            feature_to_drop = non_key_features.idxmax()
            current_df = current_df.drop(columns=[feature_to_drop])
            print(f"Feature removed: {feature_to_drop}")
        elif not missing_percent_subjects.empty:
            scelto_soggetto = False
            while not scelto_soggetto:
                subject_to_drop = missing_percent_subjects.idxmax()
                temp_df = current_df.drop(index=[subject_to_drop])
            
             # Vrify balancing after removing feature/subject
                current_balance = temp_df[balance_column].value_counts(normalize=True)
                if all(abs(initial_balance - current_balance) <= 0.2):  # Assicurarsi che il bilanciamento non cambi di più del 20%
                    current_df = temp_df
                    scelto_soggetto = True
                    print(f"Subject removed: {subject_to_drop}")
                else:
                    id = missing_percent_subjects.drop(subject_to_drop)
                    print(f"Subject not removed in order to mantain balncing: {subject_to_drop}")
            

        # Check of actual state of DataFrame
        print(f"Actual percentage of missing values: {of.calculate_missing_percentage(current_df):.2f}%")
        if current_df.shape[0] > min_subjects:
            print(f"Number of subjects remaining: {current_df.shape[0]}")
    
    return current_df




def optimization_rules(df_vercleaned,features_to_check,desired_missing_percentage):
    np.random.seed(42)

    # Copia il DataFrame ASD_phenotypic_cleaned
    cleaned_df = df_vercleaned.copy()

    while True:
        # Calcola la percentuale attuale di valori mancanti nelle features specificate
        current_missing_percentage = (cleaned_df[features_to_check].isna().sum() / len(cleaned_df)) * 100
        
        # Se la percentuale di valori mancanti è inferiore al 20%, esci dal ciclo
        if (current_missing_percentage <= desired_missing_percentage).all():
            break

        # Seleziona solo i soggetti non autistici con valori mancanti nelle due features
        non_autistic_missing = cleaned_df[(cleaned_df['DX_GROUP'] == 2) & (cleaned_df[features_to_check].isna().any(axis=1))]

        # Se non ci sono più soggetti non autistici con valori mancanti, esci dal ciclo
        if non_autistic_missing.empty:
            break
        
        
        # Seleziona un soggetto non autistico con valori mancanti in modo casuale
        subject_to_remove = np.random.choice(non_autistic_missing.index)

        # Rimuovi il soggetto selezionato
        cleaned_df = cleaned_df.drop(index=subject_to_remove)


    # Verifica il DataFrame aggiornato
    print(cleaned_df)
    return cleaned_df

