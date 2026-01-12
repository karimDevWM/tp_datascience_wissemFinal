import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os
import glob

import sys

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import importlib
import Extract
import Explore
importlib.reload(Extract)
importlib.reload(Explore)

from Extract import extract_dfs
from Explore import explore_dfs

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
else:
    st.error(f"File not found: {FILE_CLUSTER}. Please run the transformation pipeline first.")

st.header("")