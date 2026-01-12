import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def rfm_segmentation():
    print("\n--- 5. Segmentation RFM ---")

    # Charger le df_final
    try:
        df_final = pd.read_csv('dashboard/result/df_final.csv')
        print("df_final.csv chargé avec succès.")
    except FileNotFoundError:
        print("Erreur: df_final.csv non trouvé. Assurez-vous que le fichier a été généré par la fonction transform.")
        return

    # Convertir la colonne de date au format datetime
    df_final['order_purchase_timestamp'] = pd.to_datetime(df_final['order_purchase_timestamp'])
    # Rechercher la date la plus récente dans le dataset
    last_date = df_final['order_purchase_timestamp'].max()
    # Définir une date de référence pour le calcul de la Récence (par exemple, le jour après la dernière commande)
    snapshot_date = last_date + timedelta(days=1)

    # Calculer Récence, Fréquence, Montant
    rfm = df_final.groupby('customer_id').agg(
        Recency=('order_purchase_timestamp', lambda date: (snapshot_date - date.max()).days),
        Frequency=('order_id', 'count'),
        Monetary=('total_paid', 'sum')
    ).reset_index()

    # Afficher les premières lignes du tableau RFM
    print("\nTableau RFM (premières lignes) :")
    print(rfm.head())

    # Création des segments RFM 
    # Plus la récence est faible, mieux c'est, donc le score est inversé pour la récence
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    return [rfm['R_Score'], rfm['F_Score'], rfm['M_Score']]