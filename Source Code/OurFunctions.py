
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Features')


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

    


