import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt #for the plots
import seaborn as sns 
import re
import OurFunctions as of
from sklearn.preprocessing import MinMaxScaler

import missingno as msno


ASD_phenotypic_original = pd.read_csv(os.path.join('DataSets','Phenotypic Datasets','ASD_phenotypic.csv'))

# Count of the missing values
nan_values = ASD_phenotypic_original.isna().sum()
nan_sorted = nan_values.sort_values(ascending=False)

total_rows = ASD_phenotypic_original.shape[0]
percent_missing = (nan_sorted / total_rows) * 100



# We implemented a function "select_columns", that is able to define wich columns are numerical
# and which ones are categorical (also redefine the objects as categorical in the dataset)
numeric_columns, categorical_columns, ASD_phenotypic_original = of.select_columns(ASD_phenotypic_original)

# We plot the distribution of missing values, with the specification of numeric and categorical columns
of.plot_missing_values(percent_missing, numeric_columns, legend=True)

# Drop  columns DX_GROUP e DSM_IV_TR from the original ASD_phenotypic 
ASD_phenotypic = ASD_phenotypic_original.drop(columns=['DX_GROUP', 'DSM_IV_TR'])

# Store them in a new dataset called ASD_clinical
ASD_clinical = ASD_phenotypic_original[['DX_GROUP']]


for column in ASD_phenotypic:
    
    # Replace -9999 and "-9999" with NaN
    ASD_phenotypic[column] = ASD_phenotypic[column].replace(['-9999', -9999], np.NaN)
    



# Visualizzazione dei missing values
msno.matrix(ASD_phenotypic)

# Supponiamo di avere un DataFrame chiamato ASD_phenotypic
# Ecco un esempio di come potrebbe essere il tuo DataFrame
# ASD_phenotypic = pd.read_csv('your_dataset.csv')

# Features chiave da mantenere
key_features = ['FIQ', 'VIQ', 'PIQ', 'ADI_R_VERBAL_TOTAL_BV', 'ADOS_TOTAL']

def calculate_missing_percentage(df):
    total_values = df.size
    missing_values = df.isna().sum().sum()
    return (missing_values / total_values) * 100


def remove_high_missing(df, key_features, balance_column, min_subjects=200, max_missing_percentage=10):
    current_df = df.copy()
    
    # Calcolare la proporzione iniziale del bilanciamento
    initial_balance = current_df[balance_column].value_counts(normalize=True)
    
    while (calculate_missing_percentage(current_df) > max_missing_percentage) or (current_df.shape[0] < min_subjects):
        # Calcola la percentuale di valori mancanti per ciascuna colonna e riga
        missing_percent_features = current_df.isna().mean() * 100
        missing_percent_subjects = current_df.isna().mean(axis=1) * 100
        
        # Filtra le features chiave per non rimuoverle
        non_key_features = missing_percent_features.drop(labels=key_features, errors='ignore')
        
        # Trova la feature o il soggetto con il più alto tasso di missing values
        max_feature_missing = non_key_features.max()
        max_subject_missing = missing_percent_subjects.max()
        
        # Rimuovi la feature o il soggetto con il tasso di missing values più alto
        if max_feature_missing >= max_subject_missing and not non_key_features.empty:
            feature_to_drop = non_key_features.idxmax()
            current_df = current_df.drop(columns=[feature_to_drop])
            print(f"Rimosso feature: {feature_to_drop}")
        elif not missing_percent_subjects.empty:
            subject_to_drop = missing_percent_subjects.idxmax()
            temp_df = current_df.drop(index=[subject_to_drop])
            
            # Verifica il bilanciamento dopo la rimozione
            current_balance = temp_df[balance_column].value_counts(normalize=True)
            if all(abs(initial_balance - current_balance) <= 0.05):  # Assicurarsi che il bilanciamento non cambi di più del 5%
                current_df = temp_df
                print(f"Rimosso soggetto: {subject_to_drop}")
            else:
                print(f"Soggetto non rimosso per mantenere il bilanciamento: {subject_to_drop}")
        else:
            break
        
        # Controllo dello stato attuale del DataFrame
        print(f"Percentuale attuale di missing values: {calculate_missing_percentage(current_df):.2f}%")
        print(f"Numero di soggetti rimanenti: {current_df.shape[0]}")
    
    return current_df


# Applica la funzione di pulizia sul DataFrame
ASD_phenotypic_cleaned = remove_high_missing(ASD_phenotypic, key_features, balance_column='DX', min_subjects=200, max_missing_percentage=10)

# Mostra le prime righe del DataFrame pulito
print(ASD_phenotypic_cleaned.head())

# Salva il DataFrame pulito
# ASD_phenotypic_cleaned.to_csv('cleaned_dataset.csv', index=False)

ASD_phenotypic_cleaned

