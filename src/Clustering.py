from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ETAPE 4 : CLUSTERING (K-MEANS)
# =============================================================================
def clustering(df_cluster):
    print("\n--- 4. Clustering ---")

    # 1. Standardisation (Indispensable pour K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # 2. Méthode du Coude (Elbow) 
    inertias = []
    K_range = range(2, 10) # On teste de 2 à 9 clusters

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # 3. K-Means Final (Disons que le coude est à k=4) 
    k_optimal = 4
    kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(X_scaled)

    # Ajout des clusters au dataframe
    df_cluster['Cluster'] = labels

    # 4. Métriques de validation
    # Note: Silhouette peut être long sur tout le dataset, on prend un échantillon
    sample_size = 10000
    if len(X_scaled) > sample_size:
        X_sample = X_scaled[:sample_size]
        labels_sample = labels[:sample_size]
    else:
        X_sample = X_scaled
        labels_sample = labels

    sil_score = silhouette_score(X_sample, labels_sample)
    db_score = davies_bouldin_score(X_sample, labels_sample)

    print(f"Score Silhouette (échantillon) : {sil_score:.3f}")
    print(f"Score Davies-Bouldin : {db_score:.3f}")

    # 5. Visualisation des clusters
    # On visualise sur deux variables importantes (ex: Prix vs Review)
    sns.scatterplot(data=df_cluster, x='price', y='review_score', hue='Cluster', palette='viridis', alpha=0.6)

    # Analyse rapide des clusters (profiling)
    print("\nProfil moyen des clusters :")
    print(df_cluster.groupby('Cluster').mean())