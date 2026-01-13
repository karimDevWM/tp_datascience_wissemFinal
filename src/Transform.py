import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# ETAPE 3 : TRANSFORMATION (ETL)
# =============================================================================
def transform(df_dic):
    print("\n--- 3. Transformation et Nettoyage ---")

    # --- A. Traitement des Commandes (Orders) ---
    orders = df_dic['orders'].copy()
    # Filtrer sur 'delivered' 
    orders = orders[orders['order_status'] == 'delivered']

    # Feature Engineering Dates 
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['purchase_hour'] = orders['order_purchase_timestamp'].dt.hour
    orders['is_night'] = orders['purchase_hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)
    orders['is_weekend'] = orders['order_purchase_timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # --- B. Traitement Produits & Items ---
    # Merge Items + Products
    items = df_dic['order_items']
    products = df_dic['products']
    
    if 'product_category_name_translation' in df_dic:
        trans = df_dic['product_category_name_translation']
        products = pd.merge(products, trans, on='product_category_name', how='left')
        products['product_category_name'] = products['product_category_name_english'].fillna(products['product_category_name'])

    df_items_prod = pd.merge(items, products, on='product_id', how='left')

    # Agrégation par commande 
    # On calcule le poids moyen, le prix total, le fret total, et le nombre d'items
    agg_items = df_items_prod.groupby('order_id').agg({
        'product_weight_g': 'mean',
        'price': 'sum',
        'freight_value': 'sum',
        'order_item_id': 'count', # Nombre d'articles
        'product_category_name': 'first' # Catégorie principale
    }).reset_index()
    agg_items.rename(columns={'order_item_id': 'nb_items'}, inplace=True)

    # --- C. Traitement Reviews ---
    reviews = df_dic['order_reviews']
    # Agrégation par commande (moyenne du score si plusieurs reviews)
    agg_reviews = reviews.groupby('order_id')['review_score'].mean().reset_index()

    # --- D. Traitement Paiements ---
    payments = df_dic['order_payments']
    agg_payments = payments.groupby('order_id')['payment_value'].sum().reset_index()
    agg_payments.rename(columns={'payment_value': 'total_paid'}, inplace=True)

    # --- E. FUSION FINALE (MERGE)  ---
    # On part des commandes filtrées
    df_final = pd.merge(orders, agg_items, on='order_id', how='inner')
    df_final = pd.merge(df_final, agg_reviews, on='order_id', how='left')
    df_final = pd.merge(df_final, agg_payments, on='order_id', how='left')
                                                            
    # Nettoyage des NaN post-merge
    df_final['review_score'] = df_final['review_score'].fillna(df_final['review_score'].mean())
    # df_final.dropna(subset=['product_weight_g'], inplace=True) # On supprime les lignes sans poids
    # Impute missing (NaN) and zero product_weights_g with the mean of the respective product_category_name.
    # Fallback to the global mean if category mean is unavailable.
    category_mean_weights = df_final.groupby('product_category_name')['product_weight_g'].transform(
        lambda x: x[(x.notna()) & (x != 0)].mean()
    )
    global_mean_weight = df_final.loc[df_final['product_weight_g'] != 0, 'product_weight_g'].mean()
    mask_to_impute = df_final['product_weight_g'].isna() | (df_final['product_weight_g'] == 0)
    df_final.loc[mask_to_impute, 'product_weight_g'] = category_mean_weights.fillna(global_mean_weight)
    
    
    # Nettoyage final des NaN
    # 1. Remplacer les NaN dans les reviews par la moyenne (si ce n'est pas déjà fait)
    # nb_na = df_final['review_score'].isna().sum()
    # if nb_na > 0:
    #     mean_val = df_final['review_score'].mean()
    #     print(f"Imputation de {nb_na} valeurs manquantes dans 'review_score' avec la moyenne: {mean_val:.4f}")
    #     df_final['review_score'] = df_final['review_score'].fillna(mean_val)
    
    # 2. Pour les autres données physiques (prix, poids), on ne peut pas inventer.
    # donc On supprime les lignes qui ont encore des NaN.
    print(f"Taille avant nettoyage final : {df_final.shape}")
    df_final.dropna(inplace=True) 
    print(f"Taille après nettoyage final : {df_final.shape}")

    # Vérification de sécurité (Doit afficher 0)
    print(f"Nombre de NaN restants : {df_final.isna().sum().sum()}")
    
    # Export CSV
    folder_path = Path('result')
    folder_path.mkdir(parents=True, exist_ok=True)
    df_final.to_csv('dashboard/result/df_final.csv', index=False)
    
    print("Fichier df_final.csv sauvegardé.")

    # Sélection des colonnes pertinentes pour le clustering

    cols_to_keep = [
        'price', 'freight_value', 'nb_items', 'product_weight_g', 
        'review_score', 'total_paid', 'is_night', 'is_weekend'
    ]
    df_cluster = df_final[cols_to_keep].copy()

    print(f"Dataset final prêt. Shape : {df_cluster.shape}")

    # Export CSV 
    df_cluster.to_csv('dashboard/result/df_cluster.csv', index=False)
    print("Fichier df_cluster.csv sauvegardé.")
    
    return df_cluster