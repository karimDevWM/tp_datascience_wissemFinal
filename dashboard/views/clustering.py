# Create clustering.py inside the dashboard/views folder.
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import random
def render_clustering_section(df_cluster):
    st.header("Cluster Dataset (df_cluster)")
    st.write(f"Shape: {df_cluster.shape}")
    with st.expander("Show Data Preview"):
        st.dataframe(df_cluster.head())
    st.subheader("Correlation Heatmap")
    numeric_df = df_cluster.select_dtypes(include=['number'])
    if not numeric_df.empty:
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap='Blues', annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig_corr)
    # Variable Distributions
    st.subheader("Variable Distributions")
    selected_col = st.selectbox("Select a column to visualize distribution",
           numeric_df.columns)
    if selected_col:
        fig_hist, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(numeric_df[selected_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig_hist)
    # Elbow Method
    st.subheader("Clustering Optimization: Elbow Method")
    st.markdown("The **Elbow Method** helps identify the optimal number of clusters ($k$)")
    if st.button("Run Elbow Analysis"):
        with st.spinner("Computing K-Means for k=2 to 10..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_df)
            inertias = []
            K_range = range(2, 11)
            progress_bar = st.progress(0)
            for i, k in enumerate(K_range):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
                progress_bar.progress((i + 1) / len(K_range))
            fig_elbow, ax = plt.subplots(figsize=(10, 6))
            ax.plot(K_range, inertias, marker='o', linestyle='--', color='b')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method For Optimal k')
            ax.grid(True)
            st.pyplot(fig_elbow)
            plt.close(fig_elbow)
    # Validation & Profiling
    st.subheader("Clustering Validation & Profiling")
    k_selected = st.slider("Select number of clusters (k)", min_value=2, max_value=10,
           value=4)
    if st.button(f"Run Clustering (k={k_selected})"):
        with st.spinner(f"Clustering with k={k_selected}..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_df)
            kmeans_final = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
            labels = kmeans_final.fit_predict(X_scaled)
            # Metrics (Sampled if large)
            if len(X_scaled) > 10000:
                indices = list(range(len(X_scaled)))
                sample_indices = random.sample(indices, 10000)
                X_sample = X_scaled[sample_indices]
                labels_sample = labels[sample_indices]
                st.info(f"Calculating Silhouette Score on a sample of 10,000 points...")
            else:
                X_sample = X_scaled
                labels_sample = labels
            sil_score = silhouette_score(X_sample, labels_sample)
            db_score = davies_bouldin_score(X_sample, labels_sample)
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", f"{sil_score:.3f}")
            col2.metric("Davies-Bouldin Score", f"{db_score:.3f}")
            # Visualization
            df_viz = df_cluster.copy()
            df_viz['Cluster'] = labels
            st.subheader("Cluster Visualization")
            if 'price' in df_viz.columns and 'review_score' in df_viz.columns:
                fig_scatter, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_viz, x='price', y='review_score', hue='Cluster',
                       palette='viridis', alpha=0.6, ax=ax)
                ax.set_title(f"Clusters: Price vs Review Score (k={k_selected})")
                st.pyplot(fig_scatter)
            else:
                cols_to_plot = numeric_df.columns[:3].tolist()
                if cols_to_plot:
                    fig_pair = sns.pairplot(df_viz, vars=cols_to_plot, hue='Cluster',
                           palette='viridis')
                    st.pyplot(fig_pair)
            st.subheader("Cluster Profiles (Mean Values)")
            cluster_means = df_viz.groupby('Cluster')[numeric_df.columns].mean()
            st.dataframe(cluster_means.style.highlight_max(axis=0))
    else:
        st.info("No numeric columns found for correlation or clustering.")