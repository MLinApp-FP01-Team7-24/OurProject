import os
import pandas as pd
import matplotlib.pyplot as plt
def print_verbose_info(normal_path, collision_path):
    """
    Stampa informazioni dettagliate sui dati caricati dalle directory specificate.

    Args:
        normal_path (str): Percorso della directory dei dati normali.
        collision_path (str): Percorso della directory dei dati di collisione.
    """
    print("\nInformazioni sui dati normali:")
    normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv') or f.endswith('.metadata')]
    for file in normal_files:
        print(f"\nFile: {file}")
        file_path = os.path.join(normal_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
            print(df.info())
            print(df.describe())
        elif file.endswith('.metadata'):
            with open(file_path, 'r') as f:
                print(f.read())

    print("\nInformazioni sui dati di collisione:")
    collision_files = [f for f in os.listdir(collision_path) if f.endswith('.csv') or f.endswith('.metadata')]
    for file in collision_files:
        print(f"\nFile: {file}")
        file_path = os.path.join(collision_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
            print(df.info())
            print(df.describe())
        elif file.endswith('.metadata'):
            with open(file_path, 'r') as f:
                print(f.read())

def print_verbose_dataset_normal(file_path="dataset/Kuka_v1/normal/dataset.csv"):
    # Carica il dataset
    df = pd.read_csv(file_path)

    # Mostra se la colonna 'label' è presente nel DataFrame
    print("Presence of 'label' column:", 'label' in df.columns)

    # Configurazioni per visualizzare tutte le colonne e più righe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    # Stampa tutti i nomi delle colonne
    print("Column names in the dataset:")
    print(df.columns)

    # Stampa le prime 5 righe del DataFrame
    print("First 5 rows of the dataset:")
    print(df.head())

    # Filtraggio delle righe dove la label è diversa da zero
    filtered_data = df[df['label'] != 0]

    # Conversione delle colonne 'start' e 'end' in datetime
    filtered_data['start'] = pd.to_datetime(filtered_data['start'])
    filtered_data['end'] = pd.to_datetime(filtered_data['end'])

    # Calcolo della durata in secondi
    filtered_data['duration'] = (filtered_data['end'] - filtered_data['start']).dt.total_seconds()

    # Stampa i dati filtrati con label diversa da zero
    print("Filtered data where 'label' is not 0:")
    print(filtered_data[['start', 'end', 'duration', 'label']])

    # Esportazione dei dati filtrati in un nuovo file CSV
    filtered_data.to_csv('filtered_data.csv', index=False)

    # Reset delle opzioni di visualizzazione per evitare effetti collaterali in altre parti del codice
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.width')


