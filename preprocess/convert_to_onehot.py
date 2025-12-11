import pandas as pd
import numpy as np
import pickle
import os
from one_hot_encoders import StaticOneHotManager,TemporalOneHotManager

def flatten_static_data(input_path, output_path):
    df_original = pd.read_pickle(input_path)
    all_flat_columns = []
    for col in df_original.columns:
        flat_cols = df_original[col].apply(pd.Series).add_prefix(f'{col}_')
        all_flat_columns.append(flat_cols)
    df_flat = pd.concat(all_flat_columns, axis=1)
    numpy_array = df_flat.to_numpy(dtype=np.float32)
    with open(output_path, 'wb') as f:
        pickle.dump(numpy_array, f)

    print(f"Original shape: {df_original.shape}")
    print(f"New flattened NumPy array shape: {numpy_array.shape}")
    print(f"Saved flattened NumPy array to {output_path}\n")
    
def flatten_temporal_data(input_path, output_path):
    df_original = pd.read_pickle(input_path)
    all_timestamps_flat = []
    for index, patient_row in df_original.iterrows():
        sequence_length = len(patient_row.iloc[0])
        for t in range(sequence_length):
            single_timestamp_parts = []
            for feature in df_original.columns:
                one_hot_vector = patient_row[feature][t]
                single_timestamp_parts.extend(one_hot_vector)
            all_timestamps_flat.append(single_timestamp_parts)
    training_data = np.array(all_timestamps_flat, dtype=np.float32)

    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"Successfully created flattened training data.")
    print(f"  - Total patients processed: {len(df_original)}")
    print(f"  - Final training data shape: {training_data.shape}")
    print(f"  - Saved to {output_path}")  

os.makedirs("Data", exist_ok=True)
# Fit One Hot Encoders for Static Categorical Data
static_df = pd.read_csv(os.path.join("real", "static_categorical.csv"))
static_manager = StaticOneHotManager()
static_manager.fit(static_df,static_df.columns)
static_manager.save()
static_one_hot = static_manager.transform(static_df,static_df.columns)
static_one_hot.to_pickle(os.path.join("Data", "static_one_hot.pkl"))

    
# ---- Flatten Static One Hot  ----
STATIC_INPUT_PATH = os.path.join("Data", "static_one_hot.pkl")
STATIC_OUTPUT_PATH = os.path.join("Data", "static_one_hot_flat.pkl")
flatten_static_data(STATIC_INPUT_PATH, STATIC_OUTPUT_PATH)




# Fit One Hot Encoders for Temporal Categorical Data
temporal_df = pd.read_csv(os.path.join("real", "temporal_categorical.csv"))
cols = temporal_df.columns[1:]
temporal_manager = TemporalOneHotManager()
temporal_manager.fit(temporal_df,cols)
temporal_manager.save()
temporal_one_hot = temporal_manager.transform(temporal_df,cols)
temporal_one_hot.to_pickle(os.path.join("Data", "temporal_one_hot.pkl"))

    
# ---- Flatten Temporal One Hot ----- 
TEMPORAL_INPUT_PATH = os.path.join("Data", "temporal_one_hot.pkl")
TEMPORAL_OUTPUT_PATH = os.path.join("Data", "temporal_one_hot_flat.pkl")
flatten_temporal_data(TEMPORAL_INPUT_PATH, TEMPORAL_OUTPUT_PATH)