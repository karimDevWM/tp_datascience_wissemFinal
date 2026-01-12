import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.Extract import extract_dfs
from src.Explore import explore_dfs

# Paths to data files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project Root is one level up from dashboard/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Data is in the root data/ folder
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Results are in the dashboard/result/ folder
RESULT_DIR = os.path.join(CURRENT_DIR, 'result')

FILE_FINAL = os.path.join(RESULT_DIR, 'df_final.csv')
FILE_CLUSTER = os.path.join(RESULT_DIR, 'df_cluster.csv')

# Page configuration
st.set_page_config(page_title="Project Dashboard", layout="wide")

st.title("Project Data Visualization")

st.header("1. Data Audit (Raw Data)")

@st.cache_data
def load_raw_data(data_dir):
    return extract_dfs(data_dir)

try:
    df_dic = load_raw_data(DATA_DIR)
    if df_dic:
        audit_results = explore_dfs(df_dic)
        for key, df in df_dic.items():
            with st.expander(f"Audit: {key}"):
                if key in audit_results:
                    res = audit_results[key]
                    st.write(f"**Shape:** {res['shape']}")
                    
                    st.write("**Detailed Metrics:**")
                    st.dataframe(res['summary'])
                    
                    st.write("**Missing Values Visualization:**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    msno.matrix(df, ax=ax, sparkline=False, fontsize=8)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.dataframe(df.head())
    else:
        st.info("No raw data found in data/ folder.")
except Exception as e:
    st.error(f"Error loading raw data: {e}")


@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# Load data
df_final = load_data(FILE_FINAL)
# Display df_final visualizations
if df_final is not None:
    st.header("Final Dataset (df_final)")
    st.write(f"Shape: {df_final.shape}")

    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download full dataset",
        data=csv,
        file_name='df_final.csv',
        mime='text/csv',
    )

    with st.expander("Show Data Preview"):
        st.dataframe(df_final.head(10))
        st.subheader("Missing Data Visualization")
        fig_missing, ax = plt.subplots(figsize=(10, 5))
        msno.matrix(df_final, ax=ax, sparkline=False)
        st.pyplot(fig_missing)
else:
    st.error(f"File not found: {FILE_FINAL}. Please run the transformation pipeline first.")        

df_cluster = load_data(FILE_CLUSTER)
# Display df_cluster visualizations
if df_cluster is not None:
    st.header("Cluster Dataset (df_cluster)")
    st.write(f"Shape: {df_cluster.shape}")
    
    with st.expander("Show Data Preview"):
        st.dataframe(df_cluster.head())

    st.subheader("Correlation Heatmap")
    # Ensure we only correlate numeric columns
    numeric_df = df_cluster.select_dtypes(include=['number'])
    if not numeric_df.empty:
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap='Blues', annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig_corr)
    else:
        st.info("No numeric columns found for correlation.")

    st.subheader("Variable Distributions")
    if not numeric_df.empty:
        selected_col = st.selectbox("Select a column to visualize distribution", numeric_df.columns)
        if selected_col:
            fig_hist, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(numeric_df[selected_col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig_hist)

    st.subheader("Clustering Optimization: Elbow Method")
    st.markdown("""
    The **Elbow Method** helps identify the optimal number of clusters ($k$) by minimizing inertia (within-cluster sum of squares).
    Look for the "elbow" point where the inertia decrease slows down significantly.
    """)
    
    if not numeric_df.empty:
        if st.button("Run Elbow Analysis"):
            with st.spinner("Computing K-Means for k=2 to 10..."):
                # 1. Standardize the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(numeric_df)

                # 2. Calculate Inertia for different k
                inertias = []
                K_range = range(2, 11)  # Test k from 2 to 10

                progress_bar = st.progress(0)
                for i, k in enumerate(K_range):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                    progress_bar.progress((i + 1) / len(K_range))

                # 3. Plot the Elbow Curve
                fig_elbow, ax = plt.subplots(figsize=(10, 6))
                ax.plot(K_range, inertias, marker='o', linestyle='--', color='b')
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method For Optimal k')
                ax.grid(True)
                st.pyplot(fig_elbow)
                plt.close(fig_elbow)

    st.subheader("Clustering Validation & Profiling")
    st.markdown("Select a number of clusters ($k$) to compute validation metrics and view cluster profiles.")
    
    k_selected = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)
    
    if st.button(f"Run Clustering (k={k_selected})"):
        if not numeric_df.empty:
            with st.spinner(f"Clustering with k={k_selected} and calculating metrics..."):
                # 1. Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(numeric_df)
                
                # 2. Fit K-Means
                kmeans_final = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
                labels = kmeans_final.fit_predict(X_scaled)
                
                # 3. Validation Metrics
                # Sampling for Silhouette if dataset is large (>10k rows)
                if len(X_scaled) > 10000:
                    indices = list(range(len(X_scaled)))
                    import random
                    sample_indices = random.sample(indices, 10000)
                    X_sample = X_scaled[sample_indices]
                    labels_sample = labels[sample_indices]
                    st.info(f"Calculating Silhouette Score on a sample of 10,000 points (Total: {len(X_scaled)})...")
                else:
                    X_sample = X_scaled
                    labels_sample = labels
                
                sil_score = silhouette_score(X_sample, labels_sample)
                db_score = davies_bouldin_score(X_sample, labels_sample)
                
                col1, col2 = st.columns(2)
                col1.metric("Silhouette Score (Higher is better)", f"{sil_score:.3f}")
                col2.metric("Davies-Bouldin Score (Lower is better)", f"{db_score:.3f}")
                
                # 4. Visualization
                # Add cluster labels to a temporary copy for plotting
                df_viz = df_cluster.copy()
                df_viz['Cluster'] = labels
                
                st.subheader("Cluster Visualization")
                # Check if specific columns exist for the requested scatterplot
                if 'price' in df_viz.columns and 'review_score' in df_viz.columns:
                    fig_scatter, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df_viz, x='price', y='review_score', hue='Cluster', palette='viridis', alpha=0.6, ax=ax)
                    ax.set_title(f"Clusters: Price vs Review Score (k={k_selected})")
                    st.pyplot(fig_scatter)
                else:
                    st.warning("Columns 'price' and 'review_score' not found for scatterplot. Showing pairplot of first 3 numeric columns.")
                    # Fallback visualization
                    cols_to_plot = numeric_df.columns[:3].tolist()
                    if cols_to_plot:
                         fig_pair = sns.pairplot(df_viz, vars=cols_to_plot, hue='Cluster', palette='viridis')
                         st.pyplot(fig_pair)

                # 5. Profiling
                st.subheader("Cluster Profiles (Mean Values)")
                cluster_means = df_viz.groupby('Cluster')[numeric_df.columns].mean()
                st.dataframe(cluster_means.style.highlight_max(axis=0))
        else:
            st.error("No numeric data available for clustering.")

    else:
        st.warning("No numeric data available for Clustering Analysis.")

else:
    st.error(f"File not found: {FILE_CLUSTER}. Please run the transformation pipeline first.")

st.header("")