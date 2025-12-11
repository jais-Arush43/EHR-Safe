import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(512,256,256), out_dim=16, dropout=0.1,
                 use_batch_norm=True, activation='lrelu', final_activation=None):
        super().__init__()
        layers = []

        activation_fn = {
            'lrelu': nn.LeakyReLU(0.01),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }[activation]

        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d,h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d = h

        layers.append(nn.Linear(d, out_dim))
        if final_activation is not None:
            layers.append(final_activation)   

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self,in_dim,hidden_dim=256,out_dim=128,depth=2,use_post_mlp=True):
        super().__init__()

        self.gru = nn.GRU(
            input_size = in_dim,
            hidden_size = hidden_dim,
            num_layers = depth,
            batch_first = True,
            dropout = 0.1
        )

        self.attn = nn.Linear(hidden_dim, 1)

        if use_post_mlp:
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim,out_dim)
            )
        else:
            self.fc = nn.Linear(hidden_dim,out_dim)

    def forward(self,x):
        # x: (batch, seq_len, in_dim)
        out, _ = self.gru(x)
        attn_scores = self.attn(out).squeeze(-1) # (batch,seq_len)
        attn_weights = torch.softmax(attn_scores,dim=1)
        context = torch.sum(out * attn_weights.unsqueeze(-1),dim=1) # (batch, hidden_dim)
        return self.fc(context)
    
    
class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=64, dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim = 512, depth = 2,
                 num_features=None, embed_dim=None, is_categorical=False):
        super().__init__()
        self.is_categorical = is_categorical
        self.fc_init_h = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, depth * hidden_dim)
        )

        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=depth,
            batch_first=True,
            dropout=0.1 if depth > 1 else 0.0
        )

        self.head_value = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim if is_categorical else num_features),
            nn.Tanh() if is_categorical else nn.Sigmoid()
        )

        self.head_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )

        self.head_mask = nn.Sequential(
            nn.Linear(hidden_dim  , num_features)
        )

    def forward(self, e, max_seq_len, mask_threshold = 0.5):
        batch_size = e.size(0)
        device = e.device
        h_0_flat = self.fc_init_h(e)
        h_0 = h_0_flat.view(self.gru.num_layers, batch_size, self.gru.hidden_size)

        tn_hat_list, u_hat_list, mask_hat_list = [], [], []
        h_t = h_0

        seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_seq_len):
            if not active_sequences.any():
                break

            gru_input = e.unsqueeze(1)
            gru_out, h_t_new = self.gru(gru_input, h_t)

            h_t = torch.where(active_sequences.view(1, -1, 1), h_t_new, h_t) 

            tn_hat_step = self.head_value(gru_out.squeeze(1))
            u_hat_step = self.head_time(gru_out.squeeze(1)) 
            mask_hat_step = self.head_mask(gru_out.squeeze(1))  

            tn_hat_list.append(tn_hat_step)
            u_hat_list.append(u_hat_step)
            mask_hat_list.append(mask_hat_step)

            seq_lengths += active_sequences.long()

            stop_condition = (torch.sigmoid(mask_hat_step) < mask_threshold).all(dim=-1)
            active_sequences = active_sequences & ~stop_condition

        tn_hat = torch.stack(tn_hat_list, dim=1)
        u_hat = torch.stack(u_hat_list, dim=1)
        mask_hat = torch.stack(mask_hat_list, dim=1)

        return tn_hat, u_hat, mask_hat, seq_lengths
    
    
class EncoderDecoder(nn.Module):
    def __init__(self,sn_dim,sce_latent_dim,tn_dim,tce_latent_dim,sc_dim,tc_dim,latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        static_input_dim = sn_dim + sce_latent_dim
        self.static_encoder = MLP(static_input_dim, out_dim=128)
        self.temporal_num_encoder = TemporalEncoder(tn_dim + 1, out_dim=256)
        self.temporal_cat_encoder = TemporalEncoder(tce_latent_dim + 1, out_dim=256)
        self.static_mask = MLP(in_dim = sn_dim + sc_dim,hidden_dims=(32,16,8),out_dim=4)
        self.temporal_num_mask_encoder = TemporalEncoder(tn_dim,hidden_dim=128,out_dim=64)
        self.temporal_cat_mask_encoder = TemporalEncoder(tc_dim,hidden_dim=128,out_dim=64)
        fusion_dim = 128 + 256 + 256 + 4 + 64 + 64
        self.fusion = FusionMLP(fusion_dim, hidden_dims=[2048, 1024], output_dim=latent_dim)
        self.static_decoder_num = MLP(latent_dim, out_dim=sn_dim,final_activation=nn.Sigmoid())
        self.static_decoder_cat = MLP(latent_dim,out_dim=sce_latent_dim,final_activation=nn.Tanh())
        self.static_decoder_mask = MLP(latent_dim, out_dim=sn_dim + sc_dim)
        self.temporal_decoder_num = TemporalDecoder(latent_dim, num_features=tn_dim, is_categorical=False)
        self.temporal_decoder_cat = TemporalDecoder(latent_dim, num_features=tc_dim, embed_dim=tce_latent_dim,is_categorical=True)

    def encode(self,sn,sc,tn,tc,un,uc,sn_mask,sc_mask,tn_mask,tc_mask):
        static_in = torch.cat([sn,sc],dim=-1)
        static_e = self.static_encoder(static_in)
        un = un.unsqueeze(-1)
        temporal_num_in = torch.cat([tn,un],dim=-1)
        temporal_num_e = self.temporal_num_encoder(temporal_num_in)
        uc = uc.unsqueeze(-1)
        temporal_cat_in = torch.cat([tc,uc],dim=-1)
        temporal_cat_e = self.temporal_cat_encoder(temporal_cat_in)
        static_mask_in = torch.cat([sn_mask,sc_mask],dim=-1)
        static_mask_e = self.static_mask(static_mask_in)
        temporal_num_mask_e = self.temporal_num_mask_encoder(tn_mask)
        temporal_cat_mask_e = self.temporal_cat_mask_encoder(tc_mask)
        e = torch.cat([static_e,temporal_num_e,temporal_cat_e,static_mask_e,temporal_num_mask_e,temporal_cat_mask_e],dim=-1)
        e = self.fusion(e)
        return e


    def decode(self, e, max_seq_len_num=100,max_seq_len_cat=300):
        sn_hat = self.static_decoder_num(e)
        sc_hat = self.static_decoder_cat(e)
        static_mask_hat = self.static_decoder_mask(e)
        sn_dim = sn_hat.shape[-1]
        sn_mask_hat = static_mask_hat[..., :sn_dim]
        sc_mask_hat = static_mask_hat[..., sn_dim:]
        tn_hat, un_hat, tn_mask_hat, seq_len_num = self.temporal_decoder_num(e, max_seq_len_num)
        tc_hat, uc_hat, tc_mask_hat, seq_len_cat = self.temporal_decoder_cat(e, max_seq_len_cat)
        return sn_hat,sc_hat,tn_hat,tc_hat,un_hat,uc_hat,sn_mask_hat,sc_mask_hat,tn_mask_hat,tc_mask_hat,seq_len_num,seq_len_cat

    def forward(self,sn,sc,tn,tc,un,uc,sn_mask,sc_mask,tn_mask,tc_mask,max_seq_len_num=None,max_seq_len_cat=None):
        e = self.encode(sn,sc,tn,tc,un,uc,sn_mask,sc_mask,tn_mask,tc_mask)
        if max_seq_len_num is None:
            max_seq_len_num = tn.size(1)
        if max_seq_len_cat is None:
            max_seq_len_cat = tc.size(1)
        return self.decode(e, max_seq_len_num,max_seq_len_cat)

    def get_encoding(self, sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, as_numpy: bool = False):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            inputs = [sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask]
            inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]
            sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask = inputs
            e = self.encode(sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask)
            if as_numpy:
                return e.detach().cpu().numpy()
            return e
    
    def generate_decoding(self, e=None, batch_size=1, max_seq_len_num=100, max_seq_len_cat=300, device="cpu", mask_threshold=0.5):
        self.eval()
        device = torch.device(device)
        with torch.no_grad():
            (sn_hat, sc_hat, tn_hat, tc_hat, un_hat, uc_hat, sn_mask_hat, sc_mask_hat, tn_mask_hat, tc_mask_hat, seq_len_num, seq_len_cat) = self.decode(e, max_seq_len_num, max_seq_len_cat)
            
            # Apply sigmoid + threshold for all masks
            sn_mask_hat = torch.sigmoid(sn_mask_hat) > mask_threshold
            sc_mask_hat = torch.sigmoid(sc_mask_hat) > mask_threshold
            tn_mask_hat = torch.sigmoid(tn_mask_hat) > mask_threshold
            tc_mask_hat = torch.sigmoid(tc_mask_hat) > mask_threshold
    
            # Slice sequences according to predicted lengths
            tn_hat = [tn_hat[i, :seq_len_num[i]] for i in range(tn_hat.size(0))]
            tc_hat = [tc_hat[i, :seq_len_cat[i]] for i in range(tc_hat.size(0))]
            un_hat = [un_hat[i, :seq_len_num[i]] for i in range(un_hat.size(0))]
            uc_hat = [uc_hat[i, :seq_len_cat[i]] for i in range(uc_hat.size(0))]
            tn_mask_hat = [tn_mask_hat[i, :seq_len_num[i]] for i in range(tn_mask_hat.size(0))]
            tc_mask_hat = [tc_mask_hat[i, :seq_len_cat[i]] for i in range(tc_mask_hat.size(0))]
    
        return (sn_hat, sc_hat, sn_mask_hat, sc_mask_hat, tn_hat, tc_hat, un_hat, uc_hat, tn_mask_hat, tc_mask_hat)

    def compute_loss(
        self, sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask,
        sn_hat, sc_hat, sn_mask_hat, sc_mask_hat, tn_hat, tc_hat, un_hat, uc_hat,
        tn_mask_hat, tc_mask_hat, pred_seq_len_num, pred_seq_len_cat, true_seq_len_num, true_seq_len_cat,
        lambda_mse=1.0, lambda_len=0.1
    ):
        losses = {}
        bce = nn.BCEWithLogitsLoss(reduction='none')
        mse = nn.MSELoss(reduction='none')
        epsilon = 1e-8
        device = sn.device
        if not torch.is_tensor(pred_seq_len_num):
            pred_seq_len_num = torch.tensor(pred_seq_len_num, device=device, dtype=torch.float32)
        if pred_seq_len_num.dim() == 0:
            pred_seq_len_num = pred_seq_len_num.unsqueeze(0)
        if not torch.is_tensor(pred_seq_len_cat):
            pred_seq_len_cat = torch.tensor(pred_seq_len_cat, device=device, dtype=torch.float32)
        if pred_seq_len_cat.dim() == 0:
            pred_seq_len_cat = pred_seq_len_cat.unsqueeze(0)
        if not torch.is_tensor(true_seq_len_num):
            true_seq_len_num = torch.tensor(true_seq_len_num, device=device, dtype=torch.float32)
        if true_seq_len_num.dim() == 0:
            true_seq_len_num = true_seq_len_num.unsqueeze(0)
        if not torch.is_tensor(true_seq_len_cat):
            true_seq_len_cat = torch.tensor(true_seq_len_cat, device=device, dtype=torch.float32)
        if true_seq_len_cat.dim() == 0:
            true_seq_len_cat = true_seq_len_cat.unsqueeze(0)
        pred_len_num = tn_hat.size(1)
        pred_len_cat = tc_hat.size(1)
        tn_sliced = tn[:, :pred_len_num, :]
        un_sliced = un[:, :pred_len_num].unsqueeze(-1)
        tn_mask_sliced = tn_mask[:, :pred_len_num, :]
        tc_sliced = tc[:, :pred_len_cat, :]
        uc_sliced = uc[:, :pred_len_cat].unsqueeze(-1)
        tc_mask_sliced = tc_mask[:, :pred_len_cat, :]
        seq_mask_num = torch.arange(pred_len_num, device=device).unsqueeze(0) < pred_seq_len_num.unsqueeze(1)
        seq_mask_num = seq_mask_num.unsqueeze(-1).float()
        seq_mask_cat = torch.arange(pred_len_cat, device=device).unsqueeze(0) < pred_seq_len_cat.unsqueeze(1)
        seq_mask_cat = seq_mask_cat.unsqueeze(-1).float()
        losses["sn_mask"] = bce(sn_mask_hat, sn_mask).mean()
        losses["sc_mask"] = bce(sc_mask_hat, sc_mask).mean()
        tn_mask_loss = bce(tn_mask_hat, tn_mask_sliced).mean(dim=-1)
        losses["tn_mask"] = (tn_mask_loss * seq_mask_num[..., 0]).sum() / (seq_mask_num[..., 0].sum() + epsilon)
        tc_mask_loss = bce(tc_mask_hat, tc_mask_sliced).mean(dim=-1)
        losses["tc_mask"] = (tc_mask_loss * seq_mask_cat[..., 0]).sum() / (seq_mask_cat[..., 0].sum() + epsilon)
        losses["sn"] = (mse(sn_hat, sn) * sn_mask).sum() / (sn_mask.sum() + epsilon)
        losses["sc"] = mse(sc_hat, sc).mean()
        losses["tn"] = (mse(tn_hat, tn_sliced) * tn_mask_sliced * seq_mask_num).sum() / ((tn_mask_sliced * seq_mask_num).sum() + epsilon)
        losses["tc"] = (mse(tc_hat, tc_sliced) * seq_mask_cat).sum() / (seq_mask_cat.sum() + epsilon)
        losses["un"] = (mse(un_hat, un_sliced) * seq_mask_num).sum() / (seq_mask_num.sum() + epsilon)
        losses["uc"] = (mse(uc_hat, uc_sliced) * seq_mask_cat).sum() / (seq_mask_cat.sum() + epsilon)
        losses["len_num"] = F.mse_loss(pred_seq_len_num.float(), true_seq_len_num.float())
        losses["len_cat"] = F.mse_loss(pred_seq_len_cat.float(), true_seq_len_cat.float())
        total_loss = (
            losses["sn_mask"] + losses["sc_mask"] +
            losses["tn_mask"] + losses["tc_mask"] +
            lambda_mse * (losses["sn"] + losses["sc"] + losses["tn"] + losses["tc"] + losses["un"] + losses["uc"]) +
            lambda_len * (losses["len_num"] + losses["len_cat"])
        )
        return total_loss, losses

    def fit(self, train_dataloader, val_dataloader=None, epochs=20, lr=1e-3, optimizer="adam", 
            lambda_mse=1.0, lambda_len=0.1, device="cpu",
            scheduler_patience=5, scheduler_factor=0.1,resume_from=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"Training on: {device}")
        if optimizer.lower() == "adam":
            opt = optim.Adam(self.parameters(), lr=lr)
        elif optimizer.lower() == "rmsprop":
            opt = optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer must be 'adam' or 'rmsprop'")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=scheduler_patience, factor=scheduler_factor)
        start_epoch = 1
        best_val_loss = float('inf')
        
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from, optimizer=opt, scheduler=scheduler)  

        for epoch in range(start_epoch, epochs + 1):
            self.train()
            total_train_loss = 0.0
            train_loss_components = {}
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
                sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, true_seq_len_num, true_seq_len_cat = [
                    x.to(device) if torch.is_tensor(x) else x for x in batch
                ]
                max_seq_len_num = tn.size(1)
                max_seq_len_cat = tc.size(1)
                (sn_hat,sc_hat,tn_hat,tc_hat,un_hat,uc_hat,sn_mask_hat,sc_mask_hat,tn_mask_hat,tc_mask_hat,pred_seq_len_num,pred_seq_len_cat) = self(
                    sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, max_seq_len_num, max_seq_len_cat
                )
                loss, losses_dict = self.compute_loss(
                    sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask,
                    sn_hat, sc_hat, sn_mask_hat, sc_mask_hat, tn_hat, tc_hat, un_hat, uc_hat,
                    tn_mask_hat, tc_mask_hat, pred_seq_len_num, pred_seq_len_cat,
                    true_seq_len_num, true_seq_len_cat, lambda_mse=lambda_mse, lambda_len=lambda_len
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_train_loss += loss.item()
                for k, v in losses_dict.items():
                    train_loss_components[k] = train_loss_components.get(k, 0.0) + v.item()
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_components = {k: v / len(train_dataloader) for k, v in train_loss_components.items()}
            avg_val_loss = None
            val_loss_components = {}
            if val_dataloader is not None:
                self.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                        sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, true_seq_len_num, true_seq_len_cat = [
                            x.to(device) if torch.is_tensor(x) else x for x in batch
                        ]
                        max_seq_len_num = tn.size(1)
                        max_seq_len_cat = tc.size(1)
                        (sn_hat,sc_hat,tn_hat,tc_hat,un_hat,uc_hat,sn_mask_hat,sc_mask_hat,tn_mask_hat,tc_mask_hat,pred_seq_len_num,pred_seq_len_cat) = self(
                            sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask, max_seq_len_num, max_seq_len_cat)
                        loss, losses_dict = self.compute_loss(
                            sn, sc, tn, tc, un, uc, sn_mask, sc_mask, tn_mask, tc_mask,
                            sn_hat, sc_hat, sn_mask_hat, sc_mask_hat, tn_hat, tc_hat, un_hat, uc_hat,
                            tn_mask_hat, tc_mask_hat, pred_seq_len_num, pred_seq_len_cat,
                            true_seq_len_num, true_seq_len_cat, lambda_mse=lambda_mse, lambda_len=lambda_len
                        )
                        total_val_loss += loss.item()
                        for k, v in losses_dict.items():
                            val_loss_components[k] = val_loss_components.get(k, 0.0) + v.item()
                avg_val_loss = total_val_loss / len(val_dataloader)
                avg_val_components = {k: v / len(val_dataloader) for k, v in val_loss_components.items()}
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_checkpoint(opt, scheduler, epoch, filename='weights/best_encoder_decoder_ckpt.pt')
                    
            if avg_val_loss is not None:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.7f} | Val Loss: {avg_val_loss:.7f} | Train LenNum: {avg_train_components['len_num']:.6f} | Train LenCat: {avg_train_components['len_cat']:.6f} | Val LenNum: {avg_val_components['len_num']:.6f} | Val LenCat: {avg_val_components['len_cat']:.6f}")
            else:
                scheduler.step(avg_train_loss)
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.7f} | Train LenNum: {avg_train_components['len_num']:.6f} | Train LenCat: {avg_train_components['len_cat']:.6f}")

        self.save_checkpoint(opt, scheduler, epoch)

    def save_checkpoint(self, optimizer, scheduler, epoch, filename="weights/encoder_decoder_ckpt.pt"):
        checkpoint = {
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at epoch {epoch} -> {filename}")

    def load_checkpoint(self, filename="weights/encoder_decoder_ckpt.pt", optimizer=None, scheduler=None, map_location=None):
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_state_dict(checkpoint["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint {filename}, starting at epoch {start_epoch}")
        return start_epoch
