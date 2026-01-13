# Create a folder named views inside dashboard, then create audit.py.

import streamlit as st
import matplotlib.pyplot as plt
import missingno as msno
import sys
import os

# Ensure imports work if run independently or via app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import DATA_DIR
from data_loader import load_raw_data
from src.Explore import explore_dfs


def render_audit_section():
    st.header("1. Data Audit (Raw Data)")
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
