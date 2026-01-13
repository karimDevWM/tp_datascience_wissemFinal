# Create rfm.py inside the dashboard/views folder.

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import timedelta


def render_rfm_section(df_final):
    # --- Final Dataset Preview ---
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

    # --- RFM Logic ---
    st.header("RFM Segmentation")
    st.markdown("Calculating Recency, Frequency, and Monetary (RFM) scores.")

    try:
        # 1. Prepare data
        df_rfm_calc = df_final.copy()
        required_cols = ['order_purchase_timestamp', 'customer_id', 'order_id', 'total_paid']
        missing_cols = [c for c in required_cols if c not in df_rfm_calc.columns]

        if not missing_cols:
            df_rfm_calc['order_purchase_timestamp'] = pd.to_datetime(df_rfm_calc['order_purchase_timestamp'])

            last_date = df_rfm_calc['order_purchase_timestamp'].max()
            snapshot_date = last_date + timedelta(days=1)

            # 2. Calculate RFM metrics
            rfm = df_rfm_calc.groupby('customer_id').agg(
                Recency=('order_purchase_timestamp', lambda date: (snapshot_date - date.max()).days),
                Frequency=('order_id', 'count'),
                Monetary=('total_paid', 'sum')
            ).reset_index()

            # 3. Calculate Scores
            rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

            # Handle potential duplicates in Frequency/Monetary
            try:
                rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
            except ValueError:
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

            try:
                rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
            except ValueError:
                rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

            st.write("**RFM Table (First 5 rows):**")
            st.dataframe(rfm.head())

            st.subheader("RFM Distributions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Recency**")
                fig_r, ax_r = plt.subplots(figsize=(6, 4))
                sns.histplot(rfm['Recency'], kde=True, ax=ax_r)
                st.pyplot(fig_r)
            with col2:
                st.markdown("**Frequency**")
                fig_f, ax_f = plt.subplots(figsize=(6, 4))
                sns.histplot(rfm['Frequency'], kde=True, ax=ax_f)
                st.pyplot(fig_f)
            with col3:
                st.markdown("**Monetary**")
                fig_m, ax_m = plt.subplots(figsize=(6, 4))
                sns.histplot(rfm['Monetary'], kde=True, ax=ax_m)
                st.pyplot(fig_m)

        else:
            st.error(f"Required columns for RFM not found in df_final: {missing_cols}")
    except Exception as e:
        st.error(f"Error computing RFM: {e}")
