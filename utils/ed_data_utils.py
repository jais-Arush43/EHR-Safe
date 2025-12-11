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


def process_static_num(patient_row, feature_cols):
    values = []
    masks = []

    for col in feature_cols:
        val = patient_row[col]
        if pd.isna(val):
            values.append(0.0)
            masks.append(0.0)
        else:
            values.append(val)
            masks.append(1.0)

    sn = torch.tensor(values, dtype=torch.float32)     
    sn_mask = torch.tensor(masks, dtype=torch.float32) 

    return sn, sn_mask

def process_temporal_num(patient_row, feature_cols, date_col="DATE"):
    seqs = []
    masks = []

    for col in feature_cols:
        values = patient_row[col]
        if isinstance(values, str):
            values = ast.literal_eval(values)

        arr = np.array(values, dtype=np.float32)
        mask = ~np.isnan(arr)
        arr[np.isnan(arr)] = 0

        seqs.append(arr)
        masks.append(mask.astype(np.float32))
    seqs = np.stack(seqs, axis=-1)
    masks = np.stack(masks, axis=-1)
    timestamps = patient_row[date_col]
    if isinstance(timestamps, str):
        timestamps = ast.literal_eval(timestamps)
    timestamps = np.array(timestamps, dtype=np.float32)
    tn = torch.tensor(seqs, dtype=torch.float32)          # [seq_len, num_features]
    tn_mask = torch.tensor(masks, dtype=torch.float32)    # [seq_len, num_features]
    un = torch.tensor(timestamps, dtype=torch.float32)       # [seq_len]

    return tn, tn_mask, un, tn.shape[0]

def process_static_cat(one_hot_row, model, input_dims, device="cpu"):
    if isinstance(one_hot_row, np.ndarray):
        one_hot_row = torch.tensor(one_hot_row, dtype=torch.float32, device=device)
    else:
        one_hot_row = one_hot_row.to(device)
    masks = []
    start = 0
    for dim in input_dims:
        segment = one_hot_row[start:start+dim]
        masks.append(0.0 if torch.all(segment == 0) else 1.0)
        start += dim
    sc_mask = torch.tensor(masks, dtype=torch.float32, device=device)

    with torch.no_grad():
        sc = model.encode(one_hot_row.unsqueeze(0))  
    sc = sc.squeeze(0)

    return sc, sc_mask

def split_temporal_one_hot(flat_array, seq_lens):
    sequences = []
    start = 0
    for l in seq_lens:
        seq = flat_array[start:start+l]
        sequences.append(seq)
        start += l
    return sequences

def process_temporal_cat(flat_seq_list, input_dims, model, device="cuda"):
    tc_list = []
    tc_mask_list = []
    seq_len_list = []

    model.to(device)
    model.eval()

    for seq in tqdm(flat_seq_list):
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        seq_len_list.append(seq_tensor.shape[0])

        masks = []
        start = 0
        for dim in input_dims:
            segment = seq_tensor[:, start:start+dim]
            mask_segment = (segment.abs().sum(dim=1) != 0).float().unsqueeze(1)
            masks.append(mask_segment)
            start += dim
        masks = torch.cat(masks, dim=1)
        tc_mask_list.append(masks)

        with torch.no_grad():
            latent_seq = model.encode(seq_tensor)
        tc_list.append(latent_seq)

    return tc_list, tc_mask_list, seq_len_list


class EncoderDecoderDataset(Dataset):
    def __init__(self, static_num_tensor, static_num_mask_tensor,
                 static_cat_list, static_cat_mask_list,
                 tn_list, tn_mask_list, un_list,
                 tc_list, tc_mask_list, uc_list):
        self.static_num_tensor = static_num_tensor.cpu()
        self.static_num_mask_tensor = static_num_mask_tensor.cpu()
        self.static_cat_list = [t.cpu() for t in static_cat_list]
        self.static_cat_mask_list = [t.cpu() for t in static_cat_mask_list]
        self.tn_list = [t.cpu() for t in tn_list]
        self.tn_mask_list = [t.cpu() for t in tn_mask_list]
        self.un_list = [torch.as_tensor(u, dtype=torch.float32, device="cpu") for u in un_list]
        self.tc_list = [t.cpu() for t in tc_list]
        self.tc_mask_list = [t.cpu() for t in tc_mask_list]
        self.uc_list = [torch.as_tensor(u, dtype=torch.float32, device="cpu") for u in uc_list]

        self.num_patients = len(self.static_num_tensor)

    def __len__(self):
        return self.num_patients

    def __getitem__(self, idx):
        sn = self.static_num_tensor[idx]
        sn_mask = self.static_num_mask_tensor[idx]
        sc = self.static_cat_list[idx]
        sc_mask = self.static_cat_mask_list[idx]
        tn = self.tn_list[idx]
        tn_mask = self.tn_mask_list[idx]
        un = self.un_list[idx]
        tc = self.tc_list[idx]
        tc_mask = self.tc_mask_list[idx]
        uc = self.uc_list[idx]

        seq_len_num = tn.size(0)
        seq_len_cat = tc.size(0)

        return (
            sn, sc, tn, tc, un, uc,
            sn_mask, sc_mask, tn_mask, tc_mask,
            seq_len_num, seq_len_cat
        )
        
        
def collate_fn(batch):
    (sn_list, sc_list, 
     tn_list, tc_list, 
     un_list, uc_list, 
     sn_mask_list, sc_mask_list, 
     tn_mask_list, tc_mask_list, 
     seq_len_num_list, seq_len_cat_list) = zip(*batch)

    sn = torch.stack(sn_list, dim=0)
    sc = torch.stack(sc_list, dim=0)
    sn_mask = torch.stack(sn_mask_list, dim=0)
    sc_mask = torch.stack(sc_mask_list, dim=0)

    tn = pad_sequence(tn_list, batch_first=True, padding_value=0.0)
    tn_mask = pad_sequence(tn_mask_list, batch_first=True, padding_value=0.0)
    un = pad_sequence(un_list, batch_first=True, padding_value=0.0)

    tc = pad_sequence(tc_list, batch_first=True, padding_value=0.0)
    tc_mask = pad_sequence(tc_mask_list, batch_first=True, padding_value=0.0)
    uc = pad_sequence(uc_list, batch_first=True, padding_value=0.0)

    seq_len_num = torch.tensor(seq_len_num_list, dtype=torch.long)
    seq_len_cat = torch.tensor(seq_len_cat_list, dtype=torch.long)

    return sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, seq_len_num, seq_len_cat

def encode_dataloader(encoder, dataloader, device=None, as_numpy=False):
    encoder.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    embeddings_list = []
    seq_len_num_list = []
    seq_len_cat_list = []

    with torch.no_grad():
        for batch in dataloader:
            sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, seq_len_num, seq_len_cat = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            e = encoder.get_encoding(
                sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, as_numpy=False
            )
            embeddings_list.append(e.cpu())

            seq_len_num_list.append(seq_len_num.cpu())
            seq_len_cat_list.append(seq_len_cat.cpu())
            
    embeddings = torch.cat(embeddings_list, dim=0)          
    seq_len_num_all = torch.cat(seq_len_num_list, dim=0)    
    seq_len_cat_all = torch.cat(seq_len_cat_list, dim=0)     

    if as_numpy:
        return embeddings.numpy(), {
            "seq_len_num": seq_len_num_all.numpy(),
            "seq_len_cat": seq_len_cat_all.numpy()
        }
    return embeddings, {"seq_len_num": seq_len_num_all, "seq_len_cat": seq_len_cat_all}
