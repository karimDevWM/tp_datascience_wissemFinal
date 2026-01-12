import glob
import pandas as pd
import os

# =============================================================================
# ETAPE 1 : EXTRACTION
# =============================================================================
def extract_dfs(data_dir="data"):
    print("--- 1. Chargement des données ---")
    # Assurez-vous que les fichiers CSV sont dans le même dossier ou ajustez le path
    search_path = os.path.join(data_dir, "*.csv")
    all_files = glob.glob(search_path)
    
    # Dictionnaire pour stocker les DataFrames
    df_dic = {}

    # Boucle de chargement et nommage propre des clés
    for filename in all_files:
        # Nettoyage du nom pour la clé (ex: 'olist_orders_dataset.csv' -> 'orders')
        base_name = os.path.basename(filename)
        key_name = base_name.replace('.csv', '').replace('olist_', '').replace('_dataset', '')
        df_dic[key_name] = pd.read_csv(filename)
        print(f"Chargé : {key_name}")
    
    return df_dic