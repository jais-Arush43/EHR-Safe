import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split,TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import math
import json
import joblib
import ast
import pickle
import os
import random
import string
from models import StochasticNormalizer,EncoderDecoder,CategoricalAutoEncoder
from utils.ed_data_utils import *

# Normalize Static Numerical Data
static_numerical = pd.read_csv(os.path.join("real", "static_numerical.csv"))
X_age = torch.tensor(static_numerical["AGE"].values, dtype=torch.float32)
static_normalizer  = StochasticNormalizer()
print('\n Normalizing Static Numerical Data....')
X_hat_age = static_normalizer.stochastic_normalize(X_age,key = "AGE")
static_normalizer.save_params(os.path.join("weights", "static_params.pt"))
static_numerical["AGE_normalized"] = X_hat_age.numpy()
print('\n Normalized Static Numerical Data')

# Normalize Temporal Numerical Data
temporal_numerical = pd.read_csv(os.path.join("real", "temporal_numerical.csv"))
for col in temporal_numerical.columns:
    temporal_numerical[col] = temporal_numerical[col].apply(lambda x: json.loads(x.replace("nan", "null")) if isinstance(x, str) else x)
    
temporal_normalizer = StochasticNormalizer()
print('\n Normalizing Temporal Numerical Data....')
for col_idx, col in enumerate(temporal_numerical.columns):
    all_values = []
    lengths_per_cell = []

    for row in temporal_numerical[col]:
        if isinstance(row, list):
            clean_vals = [v for v in row if v is not None and not pd.isna(v)]
            
            if col_idx == 0 and len(clean_vals) > 1:  
                deltas = [clean_vals[0]] + [clean_vals[i] - clean_vals[i-1] for i in range(1, len(clean_vals))]
                all_values.extend(deltas)
            else:
                all_values.extend(clean_vals)
            
            lengths_per_cell.append(len(row))
        else:
            lengths_per_cell.append(0)

    X = torch.tensor(all_values, dtype=torch.float32)
    X_hat = temporal_normalizer.stochastic_normalize(X, key=col)

    normalized_col = []
    counter = 0
    for row in temporal_numerical[col]:
        if isinstance(row, list) and len(row) > 0:
            n_valid = len([v for v in row if v is not None and not pd.isna(v)])
            norm_cell = X_hat[counter:counter+n_valid].tolist()
            counter += n_valid

            norm_cell_with_nan = []
            idx_norm = 0
            for v in row:
                if v is None or pd.isna(v):
                    norm_cell_with_nan.append(float('nan'))
                else:
                    norm_cell_with_nan.append(norm_cell[idx_norm])
                    idx_norm += 1
            normalized_col.append(norm_cell_with_nan)
        else:
            normalized_col.append([])

    temporal_numerical[col] = normalized_col

temporal_normalizer.save_params(os.path.join("weights", "temporal_params.pt"))
print('\n Normalized Temporal Numerical Data')


# Prepare Static Numerical Data
static_num_tensor = []
static_num_mask_tensor = []
print('Preparing Static Numerical Data.....')
for idx, row in static_numerical.iterrows():
    sn, sn_mask = process_static_num(row, feature_cols=['AGE_normalized'])
    static_num_tensor.append(sn)
    static_num_mask_tensor.append(sn_mask)

static_num_tensor = torch.stack(static_num_tensor, dim=0)       
static_num_mask_tensor = torch.stack(static_num_mask_tensor, dim=0)

# Prepare Temporal Numerical Data
tn_list = []
tn_mask_list = []
un_list = []
seq_len_num = []
feature_cols = temporal_numerical.columns[1:]
print('Preparing Temporal Numerical Data.....')
for idx, row in temporal_numerical.iterrows():
    tn, tn_mask, un, seq_len = process_temporal_num(row, feature_cols, date_col="DATE")
    tn_list.append(tn)
    tn_mask_list.append(tn_mask)
    un_list.append(un)
    seq_len_num.append(seq_len)
    


static_categorical = pd.read_pickle(os.path.join("Data", "static_one_hot_flat.pkl"))
temporal_categorical = pd.read_pickle(os.path.join("Data", "temporal_one_hot_flat.pkl"))

df = pd.read_csv(os.path.join("real", "temporal_categorical.csv"))
temporal_times = df[df.columns[0]]

parsed_rows = [ast.literal_eval(row) if isinstance(row, str) else row for row in temporal_times]

delta_rows = []
for row in parsed_rows:
    row = torch.tensor(row, dtype=torch.float32)
    delta = torch.empty_like(row)
    delta[0] = row[0] 
    if len(row) > 1:
        delta[1:] = row[1:] - row[:-1]  
    delta_rows.append(delta)

all_deltas = torch.cat(delta_rows)

normalizer = StochasticNormalizer()
X_hat = normalizer.stochastic_normalize(all_deltas, key="DATE")

normalized_times = []
counter = 0
for delta in delta_rows:
    length = len(delta)
    norm_row = X_hat[counter:counter+length].tolist()
    normalized_times.append(norm_row)
    counter += length

normalizer.save_params(os.path.join("weights", "categorical_times_params.pt"))
uc_list = []
for row in normalized_times:
    if isinstance(row, str):
        row = ast.literal_eval(row)
    tensor_row = torch.tensor(row, dtype=torch.float32)
    uc_list.append(tensor_row)
    
seq_lens = []
for timestamps in temporal_times:
    if isinstance(timestamps, str):
        timestamps = ast.literal_eval(timestamps)
    seq_lens.append(len(timestamps))
    
device = "cuda"
static_one_hot = pd.read_pickle(os.path.join("Data", "static_one_hot.pkl"))
static_input_dims = [len(static_one_hot.iloc[0][col]) for col in static_one_hot.columns]
static_cat_list = []
static_cat_mask_list = []
static_input_dims = [len(static_one_hot.iloc[0][col]) for col in static_one_hot.columns]
static_cat_model = CategoricalAutoEncoder(static_input_dims, filename='weights/static_categorical_encoder_decoder.pt')
static_cat_model.load_model()
static_cat_model.to(device)
static_cat_model.eval()

for vec in tqdm(static_categorical):
    sc, sc_mask = process_static_cat(vec, static_cat_model, input_dims=static_input_dims, device=device)
    static_cat_list.append(sc)
    static_cat_mask_list.append(sc_mask)
    
    
temporal_one_hot = os.path.join("Data", "temporal_one_hot.pkl")
temporal_input_dims = [len(temporal_one_hot.iloc[0][col][0]) for col in temporal_one_hot.columns]
flat_array = temporal_categorical
flat_seq_list = split_temporal_one_hot(flat_array, seq_lens)
temporal_cat_model = CategoricalAutoEncoder(temporal_input_dims, filename="weights/temporal_categorical_encoder_decoder.pt")
temporal_cat_model.load_model()
temporal_cat_model.to(device)
temporal_cat_model.eval()
tc_list, tc_mask_list, seq_len_cat = process_temporal_cat(flat_seq_list, temporal_input_dims, temporal_cat_model)





dataset = EncoderDecoderDataset(static_num_tensor, static_num_mask_tensor,
                                static_cat_list, static_cat_mask_list,
                                tn_list, tn_mask_list, un_list,
                                tc_list, tc_mask_list, uc_list)

num_patients = len(dataset)
val_ratio = 0.05
val_size = int(num_patients * val_ratio)
train_size = num_patients - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"\nData split into:")
print(f" - Training set: {train_size} patients")
print(f" - Validation set: {val_size} patients")
print("\n--- Creating DataLoaders ---")

# --- DataLoaders ---
train_loader = DataLoader(
    train_dataset,
    batch_size=128,          
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")


sn_dim = static_num_tensor.shape[1]            
sce_latent_dim = static_cat_list[0].shape[0]   
tn_dim = tn_list[0].shape[1:][-1]                
tce_latent_dim = tc_list[0].shape[1]          
sc_dim = static_cat_mask_list[0].shape[-1]
tc_dim = tc_mask_list[0].shape[-1]                      

model = EncoderDecoder(sn_dim=sn_dim,sce_latent_dim=sce_latent_dim,tn_dim=tn_dim,
        tce_latent_dim=tce_latent_dim,sc_dim=sc_dim,tc_dim=tc_dim,latent_dim=256)
model.to(device)

model.fit(train_loader, val_loader, epochs=50, lr=1e-4, optimizer="adam", lambda_mse=2.0, 
          lambda_len=1.0,device="cuda",resume_from=None)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sn_dim = static_num_tensor.shape[1]            
sce_latent_dim = static_cat_list[0].shape[0]   
tn_dim = tn_list[0].shape[1:][-1]                
tce_latent_dim = tc_list[0].shape[1]          
sc_dim = static_cat_mask_list[0].shape[-1]
tc_dim = tc_mask_list[0].shape[-1]                      
encoder = EncoderDecoder(sn_dim=sn_dim,sce_latent_dim=sce_latent_dim,tn_dim=tn_dim,
            tce_latent_dim=tce_latent_dim,sc_dim=sc_dim,tc_dim=tc_dim,latent_dim=256)
encoder.load_checkpoint(filename="weights/best_encoder_decoder_ckpt.pt")
encoder.to(device)
emb_train, info_train = encode_dataloader(encoder, train_loader, device=device)
emb_val, info_val = encode_dataloader(encoder, val_loader, device=device)

torch.save(emb_train, "Data/encoder_embeddings_train.pt")
torch.save(emb_val, "Data/encoder_embeddings_val.pt")
print("Saved embeddings: train:", emb_train.shape, " val:", emb_val.shape)