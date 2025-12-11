import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
import ast
import os
# Static One Hot Encoder

class StaticOneHotManager:
    def __init__(self):
        self.encoders = {}

    def fit(self, df, categorical_cols):
        for col in categorical_cols:
            categories = df[col].dropna().astype(str).unique().tolist()

            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                categories=[categories]
            )
            values = df[col].dropna().astype(str).values.reshape(-1, 1)
            enc.fit(values)

            self.encoders[col] = enc

    def transform(self, df, categorical_cols, return_numpy=True):
        out = pd.DataFrame()
        if "PATIENT" in df.columns:
            out["PATIENT"] = df["PATIENT"]

        for col in categorical_cols:
            enc = self.encoders[col]

            values = df[col].astype(str).values.reshape(-1, 1)
            is_na = df[col].isna().values

            encoded = enc.transform(values)

            encoded[is_na] = 0
            if return_numpy:
                out[col] = [encoded[i, :] for i in range(encoded.shape[0])]
            else:
                out[col] = encoded.tolist()

        return out

    def inverse_transform(self, series, col):
        enc = self.encoders[col]
        categories = enc.categories_[0]  
        results = []
    
        for arr in series:
            arr = np.array(arr, dtype=float)  
            if arr.sum() == 0: 
                results.append(np.nan)
            else:
                idx = arr.argmax()  
                results.append(categories[idx])
    
        return results

    def save(self, directory="weights", filename="static_encoded.pkl"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        joblib.dump(self.encoders, path)
        print(f"Saved static one hot encoders to {path}")

    def load(self, directory="weights", filename="static_encoded.pkl"):
        path = os.path.join(directory, filename)
        self.encoders = joblib.load(path)
        print(f"Loaded static one hot encoders from {path}")

        
# Temporal One Hot Encoder       
        
class TemporalOneHotManager:
    def __init__(self):
        self.encoders = {}

    def fit(self, df, categorical_cols):
        for col in categorical_cols:
            flat_values = []
            for seq in df[col]:
                if pd.isna(seq):
                    continue
                if isinstance(seq, str):
                    try:
                        seq = ast.literal_eval(seq.replace('nan', 'None'))
                    except (ValueError, SyntaxError):
                        seq = [seq]
                if not isinstance(seq, list):
                    seq = [seq]
                for v in seq:
                    if pd.notna(v):
                        flat_values.append(str(v))

            categories = pd.Series(flat_values).unique().tolist()
            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                categories=[categories]
            )
            enc.fit(np.array(categories).reshape(-1, 1))
            self.encoders[col] = enc

    def transform(self, df, categorical_cols, return_numpy=True):
        out = pd.DataFrame()
        if "PATIENT" in df.columns:
            out["PATIENT"] = df["PATIENT"]

        for col in categorical_cols:
            enc = self.encoders[col]
            cats = enc.categories_[0].tolist()
            n_cats = len(cats)

            rows = []
            for seq in df[col]:
                if pd.isna(seq):
                    seq = []
                elif isinstance(seq, str):
                    try:
                        seq = ast.literal_eval(seq.replace('nan', 'None'))
                    except (ValueError, SyntaxError):
                        seq = [seq]

                if not isinstance(seq, list):
                    seq = [seq]

                encoded_seq = []
                for v in seq:
                    if pd.isna(v):
                        if n_cats > 0:
                            encoded_seq.append([0] * n_cats)
                    else:
                        if n_cats > 0:
                            arr = enc.transform([[str(v)]]).flatten()
                            encoded_seq.append(arr.astype(int).tolist())
                rows.append(encoded_seq)
            out[col] = rows

        return out

    def inverse_transform(self, series, col):
        enc = self.encoders[col]
        cats = enc.categories_[0].tolist()
        results = []
    
        for row in series:
            arr = np.array(row, dtype=float)
            if arr.sum() == 0:
                results.append(np.nan)
            else:
                results.append(cats[arr.argmax()])
        
        return results

    def inverse_transform2(self, series, col):
        enc = self.encoders[col]
        categories = enc.categories_[0]
        results = []
        
        for idx in series:
            if pd.isna(idx) or idx == 0:
                results.append(np.nan)
            else:
                results.append(categories[int(idx)])
        
        return results

    def save(self, directory="weights", filename="temporal_encoded.pkl"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        joblib.dump(self.encoders, path)
        print(f"Saved temporal one hot encoders to {path}")

    def load(self, directory="weights", filename="temporal_encoded.pkl"):
        path = os.path.join(directory, filename)
        self.encoders = joblib.load(path)
        print(f"Loaded temporal one hot encoders from {path}")
        
        
