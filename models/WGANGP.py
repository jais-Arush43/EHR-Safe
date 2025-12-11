import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import Generator,Discriminator

def compute_mmd(x, y, sigma=None):
    if isinstance(x, list):
        x = torch.stack(x)
    if isinstance(y, list):
        y = torch.stack(y)

    x, y = x.to(torch.float32), y.to(torch.float32)

    combined = torch.cat([x, y], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    std = combined.std(dim=0, keepdim=True) + 1e-6
    x_norm = (x - mean) / std
    y_norm = (y - mean) / std
    if sigma is None:
        xy = torch.cat([x_norm, y_norm], dim=0)
        dists = torch.cdist(xy, xy, p=2)
        sigma = torch.median(dists).item()
        if sigma == 0:
            sigma = 1.0

    def gaussian_kernel(a, b, sigma):
        dist_sq = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    k_xx = gaussian_kernel(x_norm, x_norm, sigma)
    k_yy = gaussian_kernel(y_norm, y_norm, sigma)
    k_xy = gaussian_kernel(x_norm, y_norm, sigma)

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd.item()

class WGANGP:
    def __init__(self, encoder_state_dim, latent_dim=128,
                 generator_hidden_dims=None, discriminator_hidden_dims=None,
                 lr_generator=1e-4, lr_discriminator=1e-4,
                 lambda_gp=10.0, n_critic=5, device=None,
                 plateau_factor=0.5, plateau_patience=10):

        self.encoder_state_dim = encoder_state_dim
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        self.generator = Generator(encoder_state_dim, latent_dim, generator_hidden_dims).to(self.device)
        self.discriminator = Discriminator(encoder_state_dim, discriminator_hidden_dims).to(self.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr_generator, betas=(0.5, 0.9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.9))

        # ReduceLROnPlateau scheduler
        self.scheduler_G = ReduceLROnPlateau(self.optimizer_G, mode='min', factor=plateau_factor,
                                             patience=plateau_patience)
        self.scheduler_D = ReduceLROnPlateau(self.optimizer_D, mode='min', factor=plateau_factor,
                                             patience=plateau_patience)

        self.start_epoch = 1
        self.best_mmd = float("inf")

    def gradient_penalty(self, real_samples, fake_samples):
        real_samples = real_samples.to(self.device).float()
        fake_samples = fake_samples.to(self.device).float()
        batch_size = real_samples.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device).expand_as(real_samples)
        interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples.detach()
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def discriminator_loss(self, real_samples, fake_samples):
        d_real = self.discriminator(real_samples)
        d_fake = self.discriminator(fake_samples.detach())
        wasserstein_distance = d_real.mean() - d_fake.mean()
        gp = self.gradient_penalty(real_samples, fake_samples)
        d_loss = -wasserstein_distance + gp
        return d_loss, wasserstein_distance, gp

    def generator_loss(self, fake_samples):
        d_fake = self.discriminator(fake_samples)
        g_loss = -d_fake.mean()
        return g_loss

    def generate_samples(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_samples = self.generator(z).float()
        return fake_samples

    def fit(self, train_dataloader, epochs=100, resume_from=None, 
            val_dataloader=None, verbose=True):

        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from checkpoint: {resume_from}")

        self.generator.train()
        self.discriminator.train()

        history = {
            "train_d_loss": [], "train_g_loss": [], "train_wd": [], "train_gp": [],
            "val_g_loss": [], "val_wd": [], "val_mmd": []
        }

        for epoch in range(self.start_epoch, epochs + 1):
            epoch_d_loss = epoch_g_loss = epoch_wd = epoch_gp = 0.0
            batches = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{epochs}') if verbose else train_dataloader

            for i, (real_samples,) in enumerate(progress_bar):
                batches += 1
                real_samples = real_samples.to(self.device).float()
                bsz = real_samples.size(0)

                # ---- Train Discriminator ----
                self.optimizer_D.zero_grad()
                fake_samples = self.generate_samples(bsz)
                d_loss, wd, gp = self.discriminator_loss(real_samples, fake_samples)
                d_loss.backward()
                self.optimizer_D.step()

                epoch_d_loss += d_loss.item()
                epoch_wd += wd.item()
                epoch_gp += gp.item()

                # ---- Train Generator every n_critic ----
                if i % self.n_critic == 0:
                    self.optimizer_G.zero_grad()
                    fake_samples = self.generate_samples(bsz)
                    g_loss = self.generator_loss(fake_samples)
                    g_loss.backward()
                    self.optimizer_G.step()
                    epoch_g_loss += g_loss.item()

            avg_d_loss = epoch_d_loss / batches
            avg_g_loss = epoch_g_loss / max(1, (batches // self.n_critic))
            avg_wd = epoch_wd / batches
            avg_gp = epoch_gp / batches

            history["train_d_loss"].append(avg_d_loss)
            history["train_g_loss"].append(avg_g_loss)
            history["train_wd"].append(avg_wd)
            history["train_gp"].append(avg_gp)

            # ---- Validation ----
            if val_dataloader is not None:
                self.generator.eval()
                self.discriminator.eval()
                real_embeddings, fake_embeddings = [], []

                with torch.no_grad():
                    for real_samples, in val_dataloader:
                        real_samples = real_samples.to(self.device).float()
                        fake_samples = self.generate_samples(real_samples.size(0))
                        real_embeddings.append(real_samples)
                        fake_embeddings.append(fake_samples)

                real_embeddings = torch.cat(real_embeddings, dim=0)
                fake_embeddings = torch.cat(fake_embeddings, dim=0)
                current_mmd = compute_mmd(real_embeddings, fake_embeddings)
                history["val_mmd"].append(current_mmd)

                val_g_loss = val_wd = 0.0
                val_batches = 0
                with torch.no_grad():
                    for real_samples, in val_dataloader:
                        val_batches += 1
                        real_samples = real_samples.to(self.device).float()
                        bsz = real_samples.size(0)
                        fake_samples = self.generate_samples(bsz)

                        d_real = self.discriminator(real_samples)
                        d_fake = self.discriminator(fake_samples)

                        wd = d_real.mean() - d_fake.mean()
                        g_loss = self.generator_loss(fake_samples)
                        val_g_loss += g_loss.item()
                        val_wd += wd.item()

                avg_val_g_loss = val_g_loss / val_batches
                avg_val_wd = val_wd / val_batches

                history["val_g_loss"].append(avg_val_g_loss)
                history["val_wd"].append(avg_val_wd)

                if verbose:
                    print(f"[Epoch {epoch}] Train D: {avg_d_loss:.7f}, G: {avg_g_loss:.7f}, WD: {avg_wd:.7f}, GP: {avg_gp:.7f} | "
                          f"Val WD: {avg_val_wd:.7f}, MMD: {current_mmd:.7f}")

                # Save best model
                if current_mmd < self.best_mmd:
                    self.best_mmd = current_mmd
                    self.save_checkpoint("weights/best_gan.pt", epoch, history, is_best=True)

                self.generator.train()
                self.discriminator.train()

                # ---- Step scheduler using validation MMD ----
                self.scheduler_G.step(current_mmd)
                self.scheduler_D.step(current_mmd)

            else:
                if avg_wd > getattr(self, "best_wd", float("-inf")):
                    self.best_wd = avg_wd
                    self.save_checkpoint("weights/best_gan.pt", epoch, history, is_best=True)

        self.save_checkpoint("weights/final_gan.pt", epoch, history)
        print("Training completed!")
        return history

    def save_checkpoint(self, filename, epoch, history, is_best=False):
        state = {
            "epoch": epoch + 1,
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "latent_dim": self.latent_dim,
            "encoder_state_dim": self.encoder_state_dim,
            "best_mmd": self.best_mmd,
            "history": history
        }
        torch.save(state, filename)
        if is_best:
            print(f"Best model saved at {filename} (MMD: {self.best_mmd:.7f})")
        else:
            print(f"Checkpoint saved at {filename}")

    def load_checkpoint(self, filename, map_location=None):
        checkpoint = torch.load(filename, map_location=map_location or self.device)
        self.generator.load_state_dict(checkpoint["generator_state"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.start_epoch = checkpoint["epoch"]
        self.best_mmd = checkpoint.get("best_mmd", float("inf"))
        print(f"Checkpoint loaded: {filename} (resuming at epoch {self.start_epoch})")
        return checkpoint
