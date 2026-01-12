import matplotlib.pyplot as plt
import missingno as msno
import dtale
import dtale.app as dtale_app
import pandas as pd

# =============================================================================
# ETAPE 2 : EXPLORATION
# =============================================================================
def explore_dfs(df_dic):
    print("\n--- 2. Audit des données ---")
    results = {}
    for key, df in df_dic.items():
        print(f"\nAudit de : {key}")
        print(f"Shape : {df.shape}")
        
        # Calculate detailed metrics
        summary_data = {
            'Type': df.dtypes,
            'Unique': df.nunique(),
            'Missing': df.isna().sum(),
            'Missing %': (df.isna().sum() / len(df)) * 100
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Get descriptive statistics (Min, Max, Quartiles)
        # include='all' allows catching standard stats for numeric and some for objects, 
        # but we specifically want the quartiles/min/max.
        desc = df.describe(include='all').T
        
        target_cols = ['min', '25%', '50%', '75%', 'max']
        existing_cols = [c for c in target_cols if c in desc.columns]
        
        if existing_cols:
            summary_df = summary_df.join(desc[existing_cols])
            
        # On affiche juste les infos de base pour ne pas surcharger la console
        missing_count = df.isna().sum().sum()
        print(f"Valeurs manquantes totales : {missing_count}")
        
        results[key] = {
            "shape": df.shape,
            "summary": summary_df
        }
    return results