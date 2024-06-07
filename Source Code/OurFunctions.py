
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


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

'''The function plot_missing_values() is useful to plot features, devided into 
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
'''
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

    '''


# Funzione per calcolare il conteggio delle classi per DX_GROUP
def evaluate_balancing(df):
    class_counts_DX_GROUP = df['DX_GROUP'].value_counts()


    # Stampa il conteggio delle classi per DX_GROUP
    print("Conteggio delle classi per DX_GROUP:")
    print(class_counts_DX_GROUP)

    # Visualizza la distribuzione delle classi per DX_GROUP
    class_counts_DX_GROUP.plot(kind='bar', color='blue')
    plt.title('Distribuzione delle classi per DX_GROUP')
    plt.xlabel('Classe')
    plt.ylabel('Numero di campioni')
    plt.xticks(rotation=0)
    plt.show()

    # Calcola le proporzioni delle classi per DX_GROUP
    class_proportions_DX_GROUP = df['DX_GROUP'].value_counts(normalize=True)

    # Stampa le proporzioni delle classi per DX_GROUP
    print("\nProporzioni delle classi per DX_GROUP:")
    print(class_proportions_DX_GROUP)



def calculate_missing_percentage(df):
    total_values = df.size
    missing_values = df.isna().sum().sum()
    return (missing_values / total_values) * 100


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



''''This function fixs the discrepancies in the names for the same value category in an attribute
Receives a dataset as an input and return the same dataset after having solved the discrepancies'''

def discrepancies_fixer (dataset):
    
    #We make all the caracters upper for all the categorical features
    category_columns_upper = dataset.select_dtypes(include='category').apply(lambda x: x.str.upper())
    dataset[category_columns_upper.columns] = category_columns_upper

    numeric_columns, categorical_columns, dataset = select_columns(dataset)
    categorical_column_names = categorical_columns.tolist()

    def replace_categories(category):
        # for SITE_ID
        if "UCLA" in category:
            return "UCLA"
        if "LEUVEN" in category:
            return "LEUVEN"
        if "UM" in category:
            return "UM"
        
        # for FIQ_TEST_TYPE, VIQ_TEST_TYPE and PIQ_TEST_TYPE
        if "WASI" in category:
            return "WASI"
        if "WISC" in category:
            return "WISC"
        if "WAIS" in category:
            return "WAIS"
        if "DAS" in category:
            return "DAS"
        if "HAWIK" in category:
            return "HAWIK"
        if "PPVT" in category:
            return "PPVT"
        if "RAVENS" in category:
            return "RAVENS"
        
        else:
            return category


    for i in range (len(categorical_column_names)):
        dataset[categorical_column_names[i]] = dataset[categorical_column_names[i]].apply(replace_categories).astype('category')

    return dataset

''' This function make all the ranges for the test_types to same framework based on the 
information provided by the ABIDE's datasheet'''
def test_range_discrepancies_standarization (dataset):

    new_min = 50
    new_max = 160

    for i in dataset.index:

        # For FIQ
        test_type = dataset['FIQ_TEST_TYPE'][i]
        current_value = dataset['FIQ'][i]
        if (test_type == 'DAS') or (current_value <50) or (current_value > 160):
            dataset.loc[i, 'FIQ'] = (current_value - 30) / (170 - 30) * (new_max - new_min) + new_min
    
        # For VIQ
        test_type = dataset['VIQ_TEST_TYPE'][i]
        current_value = dataset['VIQ'][i]
        if (test_type == 'DAS') or (current_value <36) or (current_value > 164):
            dataset.loc[i, 'VIQ'] = (current_value - 31) / (169 - 31) * (new_max - new_min) + new_min
        elif (test_type == 'STANFORD') or (current_value <40) or (current_value > 160):
            dataset.loc[i, 'VIQ'] = (current_value - 36) / (164 - 36) * (new_max - new_min) + new_min
        elif (test_type == 'PPVT') or (current_value <50) or (current_value > 160):
            dataset.loc[i, 'VIQ'] = (current_value - 40) / (160 - 40) * (new_max - new_min) + new_min

        # For PIQ
        test_type = dataset['PIQ_TEST_TYPE'][i]
        current_value = dataset['PIQ'][i]
        if (test_type == 'DAS') or (current_value <36) or (current_value > 164):
            dataset.loc[i, 'VIQ'] = (current_value - 31) / (166 - 31) * (new_max - new_min) + new_min
        elif (test_type == 'STANFORD') or (current_value <50) or (current_value > 160):
            dataset.loc[i, 'VIQ'] = (current_value - 36) / (164 - 36) * (new_max - new_min) + new_min

    return dataset


''' This function fill the missing values of the dataset based on the particular rules
determined for each feature'''

def inpute_missing_values (dataset, dataset_original):

    # Fix discrepancies names of categories in the features
    dataset = discrepancies_fixer(dataset)

    '''For FIQ_TEST_TYPE, PIQ_TEST_TYPE, VIQ_TEST_TYPE'''
    #checking the availability of data for FIQ, PIQ, VIQ
    feature_pairs = [
        ('FIQ_TEST_TYPE', 'FIQ'),
        ('PIQ_TEST_TYPE', 'PIQ'),
        ('VIQ_TEST_TYPE', 'VIQ')]

    # for every tuple
    for test_type_col, score_col in feature_pairs:
        mode_test_type = dataset[test_type_col].mode()[0]
        dataset[test_type_col] = dataset[test_type_col].cat.add_categories("NOT_AVAILABLE")
        # for every row of the dataset
        for index, row in dataset.iterrows():
            if pd.isnull(row[test_type_col]): #if there is a value in test_type
                if not pd.isnull(row[score_col]):
                    # Substitution of missing values with mode
                    dataset.at[index, test_type_col] = mode_test_type
                elif pd.isnull(row[score_col]):
                    dataset.at[index, test_type_col] = 'NOT_AVAILABLE'
    
    '''Some adjustments to FIQ, PIQ, VIQ'''
    dataset = test_range_discrepancies_standarization (dataset)

    '''Some adjustments to ADOS_TOTAL'''
    for i in dataset["ADOS_TOTAL"].index:
        ados_comm = dataset_original["ADOS_COMM"][i]
        ados_social = dataset_original["ADOS_SOCIAL"][i]
        if not pd.isna(ados_comm) and not pd.isna(ados_social):
            dataset.loc[i, "ADOS_TOTAL"] = ados_comm + ados_social

    '''For "FIQ", "VIQ", "PIQ", "ADOS_TOTAL", "ADI_R_VERBAL_TOTAL_BV" '''

    #list of features that we want to fill
    test_score_fatures = ["FIQ", "VIQ", "PIQ", "ADOS_TOTAL", "ADI_R_VERBAL_TOTAL_BV"]

    #function to fill with the global mean or the data feature mean
    def test_score_fill (feature_value, feature_name, feature_mean):
        # We create a dictionary to store the literature mean scores
        literature_scores = {
        "FIQ": 97.34, # EEUU mean score retrieved from https://worldpopulationreview.com/country-rankings/average-iq-by-country
        "VIQ": 97.34, # EEUU mean score retrieved from https://worldpopulationreview.com/country-rankings/average-iq-by-country
        "PIQ": 97.34, # EEUU mean score retrieved from https://worldpopulationreview.com/country-rankings/average-iq-by-country
        "ADOS_TOTAL": 9.0, # autism cutoff retrieved from https://www.researchgate.net/figure/ADOS-maximum-score-and-cut-off-points-for-ASD-15_tbl1_361212648
        "ADI_R_VERBAL_TOTAL_BV": 8.0, # autism cutoff retrieved from https://www.researchgate.net/figure/Summary-statistics-for-ADI-R-domain-scores_tbl4_6709395
        }

        if pd.isna(feature_value):
            if feature_name in literature_scores:
                return literature_scores[feature_name]
            else:
                return feature_mean
        else:
            return feature_value

    #loop for filling the features   
    for feature_name in test_score_fatures:
        feature_mean = dataset[feature_name].mean()
        dataset[feature_name] = dataset[feature_name].apply(test_score_fill, args=(feature_name, feature_mean))

    return dataset


''' This function normalizes to a range [0-1] the numerical features'''
def normalization (dataset):
    numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns_names = numeric_columns.tolist()

    for column in numeric_columns_names:
        min = dataset[column].min()
        max = dataset[column].max()

        for i in dataset.index:
            current_value = dataset[column][i]
            dataset.loc[i, column] = (current_value - min) / (max - min)

    return dataset


# Example usage
# dataset = pd.read_csv('your_dataset.csv')
# normalized_dataset = normalization(dataset)


''' This function returns a version of the dataset where the categorical features
where encoded to numerical using one hot encoding'''


def plot_distributions(df):
    numeric_columns, categorical_columns, _ = select_columns(df)
    
    if numeric_columns.any():
        # Plot delle distribuzioni per le features numeriche
        num_plots = len(numeric_columns)
        num_rows = (num_plots + 3) // 4
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
        axes = axes.flatten()

        
        for i, col in enumerate(numeric_columns):
            axes[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribuzione di {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequenza')
            axes[i].grid(True)

        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    if categorical_columns.any():
        # Plot delle distribuzioni per le features categoriche
        num_plots = len(categorical_columns)
        num_rows = (num_plots + 3) // 4
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
        axes = axes.flatten()

        
        for i, col in enumerate(categorical_columns):
            df[col].value_counts().plot(kind='bar', color='skyblue', ax=axes[i])
            axes[i].set_title(f'Distribuzione di {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequenza')
            axes[i].grid(True)

        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()




