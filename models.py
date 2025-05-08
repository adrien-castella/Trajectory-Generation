import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CustomNoiseScheduler:
    def __init__(self, num_train_timesteps=1000, num_infer_timesteps=99, num_indices=None, cat_indices=None, beta_schedule="linear"):
        self.num_train_timesteps = num_train_timesteps  # Number of timesteps for training the diffusion model
        self.num_infer_timesteps = num_infer_timesteps
        self.num_indices = num_indices  # Indices of numeric features
        self.cat_indices = cat_indices  # Indices of categorical features (list of lists)

        # Initialize DDPMScheduler for numeric features (train with more timesteps)
        self.ddpm_scheduler = DDPMScheduler(beta_schedule=beta_schedule, num_train_timesteps=self.num_train_timesteps)
        self.infr_scheduler = DPMSolverMultistepScheduler.from_config(self.ddpm_scheduler.config)

        self.reset_scheduler()

    def forward_diffusion(self, x_0, t=None):
        """
        Apply forward diffusion process to data x_0 (add noise for a single random timestep).
        :param x_0: The clean data of shape (batch_size, num_steps, 14)
        :return: noisy data at a single timestep, the noise added, and the timestep
        """
        batch_size, _, _ = x_0.shape

        # Randomly sample a timestep for each element in the batch
        if t == None:
            t = torch.randint(0, self.num_train_timesteps, (batch_size,))  # Random timestep per sample

        # Apply forward diffusion to numeric features using the DDPMScheduler
        x_t = x_0.clone()  # Start with the clean data

        # Sample noise and apply it to numeric features
        num_noise = torch.randn_like(x_t[:, :, self.num_indices])  # Gaussian noise
        x_t[:, :, self.num_indices] = self.ddpm_scheduler.add_noise(x_t[:, :, self.num_indices], num_noise, t)

        # Return noisy data at timestep t, the noise, and the timestep itself
        return x_t, num_noise, t.to(x_t.device) #, i.to(x_t.device)
    
    def diffusion_generate(self, noisy_sample, noise, t):
        x_t = noisy_sample.clone()

        x_t[:, :, self.num_indices] = self.infr_scheduler.step(
            model_output=noise[:, :, self.num_indices],
            timestep=t,
            sample=x_t[:, :, self.num_indices]
        )['prev_sample']

        return x_t
    
    def reverse_diffusion(self, noisy_sample, noise, t):
        x_t = noisy_sample.clone()

        x_t[:, :, self.num_indices] = self.ddpm_scheduler.step(
            model_output=noise[:, :, self.num_indices],
            timestep=t,
            sample=x_t[:, :, self.num_indices]
        )['prev_sample']

        return x_t

    def get_reverse_timestep(self, t):
        return self.infr_scheduler.timesteps[t]
    
    def reset_scheduler(self):
        self.infr_scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = (self.dim + 1) // 2

        i = torch.arange(half_dim, device=t.device)
        emb = torch.log(torch.tensor(10000)) * (2*i / self.dim)

        t = t.unsqueeze(-1)
        emb = t / torch.exp(emb)

        emb_sin = torch.sin(emb)
        emb_cos = torch.cos(emb)

        emb = torch.stack([emb_sin, emb_cos], dim=-1)
        emb = emb.flatten(-2, -1)

        if self.dim % 2 != 0:
            emb = emb[:, :-1]
        
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = 128, time_emb_dim: int = 128):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        x_in: Tensor of shape (batch_size, input_dim)
        t: Tensor of shape (batch_size,) for timesteps
        """
        # Apply time embedding
        t_emb = self.time_mlp(self.time_embedding(t))  # (batch_size, hidden_dim)
        
        return t_emb

class TabDDPM_Transformer(nn.Module):
    def __init__(self, input_dim=14, seq_len=20, hidden_dim=256, embed_dim=256, num_layers=8, num_heads=4, dropout=0.1):
        super().__init__()

        # Project input features per timestep to embedding dim
        self.inputs = [input_dim, seq_len, hidden_dim, embed_dim, num_layers, num_heads, dropout]
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Embeddings
        self.time_embedding = TimestepEmbedding(hidden_dim=embed_dim, time_emb_dim=embed_dim)
        # self.cond_embedding = nn.Sequential(
        #     nn.Linear(2, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim)
        # )

        # Positional encoding per timestep
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, embed_dim))

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x, t):
        """
        x: (batch, seq_len, input_dim)
        t: (batch,) - diffusion timesteps
        i: (batch,) - sequence step to focus on
        """
        _, seq_len, _ = x.shape
        x = self.input_proj(x)

        # All embeddings
        pos_emb = self.pos_embedding.unsqueeze(0)
        time_emb = self.time_embedding(t)
        # cond_emb = self.cond_embedding(c)

        embedding = (time_emb).unsqueeze(1).repeat(1, seq_len, 1)

        # Broadcast conditioning over sequence
        x = x + pos_emb + embedding
        x = self.norm(x)

        x = self.transformer(x)  # (B, T, embed_dim)
        x = self.output_proj(x)  # (B, T, input_dim)
        return x

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, input_size=30):  # 10 steps * 3 features
        super(TrajectoryDiscriminator, self).__init__()
        
        # Define a simple feedforward neural network
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input to 1D
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet1DTimeSeries(nn.Module):
    def __init__(self, input_channels=1, base_channels=32, time_emb_dim=128, feature_dim=13):
        super().__init__()

        self.inputs = [input_channels, base_channels, time_emb_dim, feature_dim]

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, base_channels)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(2, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, base_channels)
        )

        # enc1 now expects input + time_emb + cond_emb concatenated => 1 + 2 = 1 + 2 * time_emb_dim channels
        self.enc1 = ConvBlock(input_channels + 2 * base_channels, base_channels, kernel_size=(3, 1), padding=(1, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        self.enc2 = ConvBlock(base_channels, base_channels * 2, kernel_size=(3, 1), padding=(1, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4, kernel_size=(3, 1), padding=(1, 0))

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=(2, 1), stride=(2, 1))
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, kernel_size=(3, 1), padding=(1, 0))

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=(2, 1), stride=(2, 1))
        self.dec1 = ConvBlock(base_channels * 2, base_channels, kernel_size=(3, 1), padding=(1, 0))

        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x, t, cond):
        # x shape: (B, T, F)
        B, H, W = x.shape
        x = x.unsqueeze(1)  # (B, 1, T, F)

        t = t.view(B, 1).float() / 1000
        t_emb = self.time_embed(t).view(B, -1, 1, 1).expand(-1, -1, H, W)
        cond_emb = self.cond_embed(cond).view(B, -1, 1, 1).expand(-1, -1, H, W)

        x = torch.cat([x, t_emb, cond_emb], dim=1)  # (B, 1 + 2*time_emb_dim, T, F)

        e1 = self.enc1(x)  # (B, base_channels, T, F)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final(d1)  # (B, 1, T, F)
        return out.squeeze(1)  # (B, T, F)