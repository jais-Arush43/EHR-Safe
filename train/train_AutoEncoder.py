import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import os
from models.CategoricalAutoEncoder import CategoricalAutoEncoder

class OneHotDataset(Dataset):
    def __init__(self, numpy_array):
        self.data = torch.from_numpy(numpy_array)
        print(f"Created dataset with shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx],)
    
STATIC_FILE_PATH = os.path.join("Data", "static_one_hot_flat.pkl")
with open(STATIC_FILE_PATH, 'rb') as f:
    static_full_dataset = pickle.load(f)

static_train_data, static_val_data = train_test_split(
    static_full_dataset,
    test_size=0.1,
    random_state=123
)

print(f"\nData split into:")
print(f" - Training set shape: {static_train_data.shape}")
print(f" - Validation set shape: {static_val_data.shape}")
print("\n--- Creating DataLoaders ---")
static_train_dataset = OneHotDataset(static_train_data)
static_val_dataset = OneHotDataset(static_val_data)

static_train_dataloader = DataLoader(dataset=static_train_dataset, batch_size=64, shuffle=True)
static_val_dataloader = DataLoader(dataset=static_val_dataset, batch_size=64, shuffle=False)

static_one_hot = pd.read_pickle(os.path.join("Data", "static_one_hot.pkl"))
static_input_dims = [len(static_one_hot.iloc[0][col]) for col in static_one_hot.columns]

# Train Static One Hot into Latent Embedding
device = "cuda"
model_instance = CategoricalAutoEncoder(static_input_dims,use_scheduler=True,filepath="weights/static_categorical_encoder_decoder.pt")
model_wrapper = nn.DataParallel(model_instance)
model_wrapper = model_wrapper.to(device)
model_wrapper.module.fit(static_train_dataloader, epochs=30, val_dataloader=static_val_dataloader,wrapper_model=model_wrapper)




TEMPORAL_FILE_PATH = os.path.join("Data", "temporal_one_hot_flat.pkl")
with open(TEMPORAL_FILE_PATH, 'rb') as f:
    temporal_full_dataset = pickle.load(f)

temporal_train_data, temporal_val_data = train_test_split(
    temporal_full_dataset,
    test_size=0.05,
    random_state=123
)

print(f"\nData split into:")
print(f" - Training set shape: {temporal_train_data.shape}")
print(f" - Validation set shape: {temporal_val_data.shape}")
print("\n--- Creating DataLoaders ---")
temporal_train_dataset = OneHotDataset(temporal_train_data)
temporal_val_dataset = OneHotDataset(temporal_val_data)

temporal_train_dataloader = DataLoader(
    temporal_train_dataset,
    batch_size=64,
    shuffle=True, 
)

temporal_val_dataloader = DataLoader(
    temporal_val_dataset,
    batch_size=64,
    shuffle=False,
)

temporal_one_hot = os.path.join("Data", "temporal_one_hot.pkl")
temporal_input_dims = [len(temporal_one_hot.iloc[0][col][0]) for col in temporal_one_hot.columns]


# Train Temporal One Hot into Latent Embedding
device = "cuda"
model_instance = CategoricalAutoEncoder(temporal_input_dims,use_scheduler=True,filepath="weights/temporal_categorical_encoder_decoder.pt")
model_wrapper = nn.DataParallel(model_instance)
model_wrapper = model_wrapper.to(device)
model_wrapper.module.fit(temporal_train_dataloader, epochs=50, val_dataloader=temporal_val_dataloader,wrapper_model=model_wrapper)





