import ast
import numpy as np
import pandas as pd

def convert_to_wide_format(long_df, time_col='TIME'):
    long_df = long_df.sort_index()
    
    time_diff = long_df[time_col].diff()
    
    new_patient_marker = (time_diff <= 0).astype(int)
    
    patient_id_col = new_patient_marker.cumsum()
    
    df_with_ids = long_df.copy()
    df_with_ids['patient_id'] = patient_id_col
    
    print("Aggregating data... this may take a moment.")
    wide_df = df_with_ids.groupby('patient_id').agg(list)
    
    if 'patient_id' in wide_df.columns:
        wide_df = wide_df.drop(columns=['patient_id'])
        
    print("Conversion complete.")
    return wide_df


def safe_list_eval(s):
    if pd.isna(s):
        return []
    if isinstance(s, (list, np.ndarray)):
        return s 

    if not isinstance(s, str) or not s.startswith('['):
        return [s] 

    try:
        s_safe = s.replace('nan', 'None') 
        evaluated_list = ast.literal_eval(s_safe)
        evaluated_list = [np.nan if x is None else x for x in evaluated_list]
        return evaluated_list
    except (ValueError, SyntaxError):
        return []

def fix_dataframe_dtypes(df):
    print(f"Fixing dtypes for {df.shape[0]} rows...")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(safe_list_eval)
            
    print("Dtype conversion complete.")
    return df

def combine_dataframes(temp_num_df, temp_cat_df, static_num_df, static_cat_df):
    
    temporal_combined = temp_num_df.join(temp_cat_df, how='outer')
    static_combined = static_num_df.join(static_cat_df, how='outer')
    
    static_cols = static_combined.columns
    for col in static_cols:
        static_combined[col] = static_combined[col].apply(
            lambda x: [x] if pd.notna(x) else []
        )
            
    final_wide_df = temporal_combined.join(static_combined, how='outer')
    
    for col in final_wide_df.columns:
        final_wide_df[col] = final_wide_df[col].apply(
            lambda x: x if isinstance(x, list) else []
        )
            
    return final_wide_df

