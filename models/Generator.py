import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,encoder_state_dim,latent_dim = 256,hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512,1024,2048,1024,512]

        layers = []
        prev_dim = latent_dim

        for i,hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2 if i < len(hidden_dims)//2 else 0.1)
            ])
            prev_dim = hidden_dim

        layers.extend([
            nn.Linear(prev_dim,encoder_state_dim*2),
            nn.BatchNorm1d(encoder_state_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(encoder_state_dim*2,encoder_state_dim)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self,z):
        return self.model(z)