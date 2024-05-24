
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd


'''The function select_columns(df) is useful for selecting and distinguishing
numerical features and categorical ones. In order to use it the framework is:

numeric_columns, categorical_columns, df = select_columns(our_dataset_inuse)'''

def select_columns(df):
    # Seleziona le colonne numeriche
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Seleziona tutte le colonne di tipo 'object'
    object_columns = df.select_dtypes(include=['object']).columns

    # Converti le colonne selezionate in tipo 'categorical'
    df[object_columns] = df[object_columns].astype('category')

    # Seleziona solo le colonne di tipo 'category'
    categorical_columns = df.select_dtypes(include=['category']).columns
    
    return numeric_columns, categorical_columns, df

'''The fanction plot_missing_values() is useful to plot features, devided into 
categorical and numerical, according to the number of missing values. In order to use it:

plot_missing_values(columns_with_nan_sorted, numeric_columns)'''


def plot_missing_values(columns_with_nan_sorted, numeric_columns, legend):
    # Assegna colori alle colonne
    '''Utilizza la lista numeric_columns per determinare se una colonna è numerica
    o meno, e assegna il colore blu alle colonne numeriche e arancione alle colonne
    non numeriche.'''
    colors = ['blue' if col in numeric_columns else 'orange' for col in columns_with_nan_sorted.index]

    # Plot delle colonne con valori mancanti
    plt.figure(figsize=(12, 16))
    columns_with_nan_sorted.plot(kind='barh', color=colors)
    plt.title('Features with Missing Values')
    plt.xlabel('Percentage of Missing Values [%]')
    plt.ylabel('Features')
    plt.xticks(range(0, 110, 10))


    # Aggiungi legenda
    
    blue_patch = mpatches.Patch(color='blue', label='Numeric Columns')
    orange_patch = mpatches.Patch(color='orange', label='Categorical Columns')
    plt.legend(handles=[blue_patch, orange_patch])

     # Se la legenda non è richiesta, non mostrare la legenda
    if not legend:
        plt.legend().remove()

    

'''stavolta riferito al numero di valori presenti
Utilizzo della funzione
plot_features_by_presence(columns_by_presence, numeric_columns)'''

def plot_features_by_presence(columns_by_presence, numeric_columns, legend):
    # Assegna colori alle colonne
    colors = ['blue' if col in numeric_columns else 'orange' for col in columns_by_presence.index]

    # Plot delle colonne con valori presenti
    plt.figure(figsize=(12, 16))
    columns_by_presence.plot(kind='barh', color=colors)
    plt.title('Features with Present Values')
    plt.xlabel('Number of Present Values')
    plt.ylabel('Features')

    # Aggiungi legenda
    
    blue_patch = mpatches.Patch(color='blue', label='Numeric Columns')
    orange_patch = mpatches.Patch(color='orange', label='Categorical Columns')
    plt.legend(handles=[blue_patch, orange_patch])

    # Se la legenda non è richiesta, non mostrare la legenda
    if not legend:
        plt.legend().remove()


'''This fucntion is used to compute the correlation between categorical variables using the cramers

extracted from chatgpt ujuju '''

def cramers_v(x, y):
    """Compute Cramer's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    total_elements = confusion_matrix.sum().sum()
    phi2 = chi2 / total_elements
    raws, columns = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((columns - 1) * (raws - 1)) / (total_elements - 1))
    raws_corr = raws - ((raws - 1) ** 2) / (total_elements - 1)
    columns_corr = columns - ((columns - 1) ** 2) / (total_elements - 1)
    return np.sqrt(phi2corr / min((columns_corr - 1), (raws_corr - 1)))

    


def calculate_missing_percentage(df):
    total_values = df.size
    missing_values = df.isna().sum().sum()
    return (missing_values / total_values) * 100


def remove_high_missing(df, threshold=10):
    current_df = df.copy()
    
    while calculate_missing_percentage(current_df) > threshold:
        # Calcola la percentuale di valori mancanti per ciascuna colonna e riga
        missing_percent_features = current_df.isna().mean() * 100
        missing_percent_subjects = current_df.isna().mean(axis=1) * 100
        
        # Trova la feature o il soggetto con il più alto tasso di missing values
        max_feature_missing = missing_percent_features.max()
        max_subject_missing = missing_percent_subjects.max()
        
        # Rimuovi la feature o il soggetto con il tasso di missing values più alto
        if max_feature_missing >= max_subject_missing:
            feature_to_drop = missing_percent_features.idxmax()
            current_df = current_df.drop(columns=[feature_to_drop])
            print(f"Rimosso feature: {feature_to_drop}")
        else:
            subject_to_drop = missing_percent_subjects.idxmax()
            current_df = current_df.drop(index=[subject_to_drop])
            print(f"Rimosso soggetto: {subject_to_drop}")
        
        # Controllo dello stato attuale del DataFrame
        print(f"Percentuale attuale di missing values: {calculate_missing_percentage(current_df):.2f}%")
    
    return current_df