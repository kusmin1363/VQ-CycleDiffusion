import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, z_dim, dec_dim, n_mels):
        super().__init__()
        self.prenet = nn.Conv1d(
            in_channels=z_dim,
            out_channels=dec_dim,
            kernel_size=3,
            padding=1
        )

        self.conv1 = nn.Conv1d(dec_dim, dec_dim, kernel_size=3, padding=1)
        self.proj = nn.Conv1d(dec_dim, n_mels, kernel_size=1)

    def forward(self, z_q):
        x = self.prenet(z_q)
        x = self.conv1(x)
        mel_hat = self.proj(x)  # → (B, 80, T)
        return mel_hat
