import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class CategoricalAutoEncoder(nn.Module):

    def __init__(self, input_dims, lr=1e-3, optimizer_type="adam", use_scheduler=False, filepath=None):
        super().__init__()

        self.input_dims = input_dims
        self.filepath = filepath
        self.total_input_dim = sum(self.input_dims)
        self.num_features = len(self.input_dims)
        hidden_dim = max(256, self.total_input_dim)
        latent_dim = max(32, min(self.total_input_dim // 8, self.total_input_dim - 1))

        # ------ Encoder ----------- 
        self.encoder = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()
        )

        # decoders: one head per feature
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.LeakyReLU(0.2),

                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(hidden_dim, k)
            )
            for k in self.input_dims
        ])

        # loss & optimizer
        self.criterion = nn.CrossEntropyLoss()
        opt_type = optimizer_type.lower()
        if opt_type == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif opt_type == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise ValueError("optimizer_type must be 'adam' or 'rmsprop'")

        self.scheduler = (optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=0.5, patience=3)
                         if use_scheduler else None)

    # ---------- Forward / Encode / Decode ----------
    def forward(self, x):
        z = self.encoder(x)
        logits_list = [head(z) for head in self.decoders]
        return logits_list

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor, mask=None, return_onehot=True):
        self.eval()
        with torch.no_grad():
            logits_list = [head(z) for head in self.decoders]
            preds = [logits.argmax(dim=-1) for logits in logits_list]

            if not return_onehot:
                return preds

            bsz = z.shape[0]
            parts = []

            for i, (idxs, K) in enumerate(zip(preds, self.input_dims)):
                onehot = torch.zeros(bsz, K, device=z.device)
                onehot.scatter_(1, idxs.view(-1, 1), 1.0)

                if mask is not None:
                    onehot = onehot * mask[:, i].unsqueeze(1)

                parts.append(onehot)

            return preds, torch.cat(parts, dim=1)

    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            logits_list = self.forward(x)
            probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]
        return probs_list

    # ---------- Reconstruct with mask for missingness ----------
    def reconstruct(self, x, return_onehot=False):
        probs_list = self.predict_prob(x)
        preds = []
        bsz = x.shape[0]
        parts = []

        start = 0
        for i, K in enumerate(self.input_dims):
            end = start + K
            segment = x[:, start:end]

            mask_all_zero = (segment.sum(dim=1) == 0)
            segment_logits = probs_list[i]
            segment_pred = segment_logits.argmax(dim=-1)
            preds.append(segment_pred)

            if return_onehot:
                onehot = torch.zeros(bsz, K, device=x.device)
                idxs_to_scatter = (~mask_all_zero).nonzero(as_tuple=True)[0]
                if len(idxs_to_scatter) > 0:
                    onehot[idxs_to_scatter].scatter_(1, segment_pred[idxs_to_scatter].view(-1, 1), 1.0)
                parts.append(onehot)

            start = end

        if return_onehot:
            return preds, torch.cat(parts, dim=1)
        return preds

    # ---------- Helper functions ----------
    def _targets_from_onehot(self, x):
        parts = torch.split(x, self.input_dims, dim=1)
        return torch.stack([p.argmax(dim=1) for p in parts], dim=1).long()

    def _compute_loss(self, logits_list, targets, mask=None):
        loss = 0.0
        for i, logits in enumerate(logits_list):
            if mask is not None:
                present_idx = (mask[:, i] == 1).nonzero(as_tuple=True)[0]
                if len(present_idx) == 0:
                    continue
                loss += self.criterion(logits[present_idx], targets[present_idx, i])
            else:
                loss += self.criterion(logits, targets[:, i])
        return loss

    # ---------- Training ----------
    def fit(self, dataloader, epochs=10, val_dataloader=None, wrapper_model=None, mask_val=None):
        forward_model = wrapper_model if wrapper_model is not None else self
        best_val_loss = float('inf')
        best_weights = None
        device = next(self.parameters()).device

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_losses = []

            for (x,) in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                x = x.to(device, non_blocking=True)
                y = self._targets_from_onehot(x)
                self.optimizer.zero_grad()
                logits_list = forward_model(x)
                loss = self._compute_loss(logits_list, y)
                if wrapper_model is not None:
                    loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = sum(epoch_losses) / max(1, len(epoch_losses))

            val_loss = None
            if val_dataloader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for i, (vx,) in enumerate(val_dataloader):
                        vx = vx.to(device, non_blocking=True)
                        vy = self._targets_from_onehot(vx)

                        # slice mask_val safely and move to device if provided
                        vmask = None
                        if mask_val is not None:
                            # mask_val may be list/np/tensor
                            if isinstance(mask_val, (list, tuple)):
                                start = i * vx.size(0)
                                vmask = torch.as_tensor(mask_val[start:start + vx.size(0)], dtype=torch.float32, device=device)
                            else:
                                m = mask_val
                                if not torch.is_tensor(m):
                                    m = torch.as_tensor(m)
                                start = i * vx.size(0)
                                vmask = m[start:start + vx.size(0)].to(device)

                        vlogits_list = forward_model(vx)
                        if vmask is not None:
                            vloss = self._compute_loss(vlogits_list, vy, mask=vmask)
                        else:
                            vloss = self._compute_loss(vlogits_list, vy)
                        val_losses.append(vloss.item())
                val_loss = sum(val_losses) / max(1, len(val_losses))

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                # Save best weights
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.state_dict()

            # print summary
            if val_loss is not None:
                print(f"Epoch {epoch:03d} | Train {train_loss:.7f} | Val {val_loss:.7f}")
            else:
                print(f"Epoch {epoch:03d} | Train {train_loss:.7f}")

        # Save best weights
        if best_weights is not None:
            if self.filepath is None:
                print("Warning: best weights found but `filepath` is None. Use `.save_model_to(path)` to save manually.")
            else:
                # ensure directory exists
                os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
                torch.save(best_weights, self.filepath)
                print(f"Best model saved with val_loss={best_val_loss:.7f} as {self.filepath}")
        else:
            # fallback: save current model if filepath provided
            if self.filepath is not None:
                self.save_model()
            else:
                print("No best weights (no validation) and no filepath provided — model not saved automatically.")

    # ---------- Save / Load ----------
    def save_model(self):
        if self.filepath is None:
            raise ValueError("`filepath` is None. Set `self.filepath` or use `save_model_to(path)`.")
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        torch.save(self.state_dict(), self.filepath)
        print(f"Model saved as {self.filepath}")

    def save_model_to(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved as {path}")

    def load_model(self, map_location=None):
        if self.filepath is None:
            raise ValueError("`filepath` is None — set `self.filepath` before calling load_model().")
        state = torch.load(self.filepath, map_location=map_location)
        self.load_state_dict(state)
        self.eval()
        print(f"Model loaded from {self.filepath}")
