import torch
from torch.utils.data import Dataset, DataLoader,random_split,TensorDataset
import os
from models import WGANGP


emb_train = torch.load(os.path.join("Data", "encoder_embeddings_train.pt"))
emb_val   = torch.load(os.path.join("Data", "encoder_embeddings_val.pt"))

all_embeddings = torch.cat([emb_train, emb_val], dim=0)
gan_dataset = TensorDataset(all_embeddings)
dataset_size = len(gan_dataset)
split_size = int(0.05 * dataset_size)
train_dataset, val_dataset = random_split(gan_dataset, [dataset_size - split_size, split_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False,num_workers=4)
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")


wgan = WGANGP(encoder_state_dim=256,latent_dim=256)
history = wgan.fit(
    train_dataloader=train_loader,   
    epochs=200,
    val_dataloader=val_loader,
    resume_from=None,
    verbose=True
)