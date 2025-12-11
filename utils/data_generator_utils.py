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

def generate_synthetic_dataset(generator, decoder, total_samples, batch_size=128, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device).eval()

    sn_list, sc_list, sn_mask_list, sc_mask_list = [], [], [], []
    tn_list, tc_list, un_list, uc_list = [], [], [], []
    tn_mask_list, tc_mask_list = [], []

    with torch.no_grad():
        for start in tqdm(range(0, total_samples, batch_size)):
            current_batch = min(batch_size, total_samples - start)
            latent_samples = generator.generate_samples(batch_size=current_batch).to(device)
            decoded = decoder.generate_decoding(latent_samples, batch_size=current_batch)
            sn_hat, sc_hat, sn_mask_hat, sc_mask_hat, tn_hat, tc_hat, un_hat, uc_hat, tn_mask_hat, tc_mask_hat = decoded

            sn_list.append(sn_hat)
            sc_list.append(sc_hat)
            sn_mask_list.append(sn_mask_hat)
            sc_mask_list.append(sc_mask_hat)

            # Extend temporal lists to preserve batch separation
            if isinstance(tn_hat, list):
                tn_list.extend(tn_hat)
            else:
                tn_list.append(tn_hat)

            if isinstance(tc_hat, list):
                tc_list.extend(tc_hat)
            else:
                tc_list.append(tc_hat)

            if isinstance(un_hat, list):
                un_list.extend(un_hat)
            else:
                un_list.append(un_hat)

            if isinstance(uc_hat, list):
                uc_list.extend(uc_hat)
            else:
                uc_list.append(uc_hat)

            if isinstance(tn_mask_hat, list):
                tn_mask_list.extend(tn_mask_hat)
            else:
                tn_mask_list.append(tn_mask_hat)

            if isinstance(tc_mask_hat, list):
                tc_mask_list.extend(tc_mask_hat)
            else:
                tc_mask_list.append(tc_mask_hat)

    return (
        torch.cat(sn_list, dim=0),     
        torch.cat(sc_list, dim=0),     
        torch.cat(sn_mask_list, dim=0),
        torch.cat(sc_mask_list, dim=0),
        tn_list,                       
        tc_list,                        
        un_list,                       
        uc_list,                       
        tn_mask_list,                    
        tc_mask_list                   
    )
    
    
def denormalize_generated_data(sn_hat, tn_hat, un_hat, sn_mask_hat, tn_mask_hat,
                               normalizer_sn, normalizer_tn, temporal_columns, use_mask=True):
    if isinstance(sn_hat, list):
        sn_hat = torch.cat(sn_hat, dim=0)

    sn = normalizer_sn.stochastic_renormalize(sn_hat, key="AGE").tolist()

    tn_flat = torch.cat(tn_hat, dim=0)
    un_flat = torch.cat(un_hat, dim=0)
    lengths = [x.shape[0] for x in tn_hat]

    tn_denorm = torch.zeros_like(tn_flat)
    for f, key in enumerate(tqdm(temporal_columns[1:], desc="Denormalizing temporal features", mininterval=0.1, leave=False)):
        tn_denorm[:, f] = normalizer_tn.stochastic_renormalize(tn_flat[:, f], key=key)

    un_denorm = normalizer_tn.stochastic_renormalize(un_flat, key=temporal_columns[0])
    un_time = []
    start = 0
    for L in tqdm(lengths, desc="Building cumulative time", mininterval=0.1, leave=False):
        end = start + L
        un_time.append(torch.cumsum(un_denorm[start:end], dim=0))
        start = end

    if use_mask:
        if sn_mask_hat is not None:
            sn_mask_hat = sn_mask_hat.squeeze() if sn_mask_hat.ndim > 1 else sn_mask_hat
            sn = [float('nan') if m == 0 else (x[0] if isinstance(x, list) else x) for x, m in zip(sn, sn_mask_hat.tolist())]
        if tn_mask_hat is not None:
            tn_mask_flat = torch.cat(tn_mask_hat, dim=0)
            tn_denorm = tn_denorm.masked_fill(tn_mask_flat == 0, float('nan'))

    static_df = pd.DataFrame({
        "AGE": sn
    })

    temporal_df = pd.DataFrame(
        tn_denorm.cpu().numpy(),
        columns=temporal_columns[1:]
    )
    temporal_df.insert(0, "TIME", torch.cat(un_time).cpu().numpy())

    return static_df, temporal_df

def denormalize_uc(uc_hat, normalizer_uc):
    if isinstance(uc_hat, list):
        lengths = [x.shape[0] for x in uc_hat]
        uc_flat = torch.cat(uc_hat, dim=0)
    else:
        lengths = [uc_hat.shape[0]]
        uc_flat = uc_hat
    uc_denorm = normalizer_uc.stochastic_renormalize(uc_flat, key='DATE')
    uc_time_list = []
    start = 0
    for L in lengths:
        end = start + L
        uc_time_list.append(torch.cumsum(uc_denorm[start:end], dim=0))
        start = end
    uc_time_flat = torch.cat(uc_time_list, dim=0)

    return uc_time_flat