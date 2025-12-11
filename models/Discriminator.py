import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,encoder_state_dim,hidden_dims = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256,512,1024,2048,1024,512,256,128]

        layers = []
        prev_dim = encoder_state_dim

        for i,hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim,hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3 if i < len(hidden_dims)//2 else 0.2)
            ])
            prev_dim = hidden_dim

        layers.extend([
            nn.Linear(prev_dim,64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64,1)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)