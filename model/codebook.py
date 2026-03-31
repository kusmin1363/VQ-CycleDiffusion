# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        # inputs: (B, D, T)
        B, D, T = inputs.shape
        # [B, D, T] → [B*T, D]
        flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, D)

        # Compute distances to embedding weights
        # distances: [B*T, K]
        embeddings = self.embeddings.weight.to(flat_input.device) 
        distances = torch.cdist(flat_input.double(), embeddings.double(), p=2)

        # Get nearest embedding index for each input vector
        encoding_indices = torch.argmin(distances, dim=1)  # [B*T]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)
        # print("encoding_indices:", encoding_indices)
        
        # Quantize
        quantized = embeddings[encoding_indices]  # [B*T, D]
        quantized = quantized.view(B, T, D).permute(0, 2, 1)  # → [B, D, T]


        # Compute loss
        quantized_detached = quantized.detach()
        e_latent_loss = F.mse_loss(quantized_detached, inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity (optional)
        avg_probs = encodings.mean(dim=0)

        return quantized, loss, e_latent_loss, q_latent_loss, encoding_indices

# VectorQuantizer 클래스 안에 이 메서드를 추가해야 합니다.
    def get_code_indices(self, inputs):
        """
        forward_pass와 동일한 로직으로, 인덱스만 반환합니다.
        """
        # inputs: (B, D, T)
        B, D, T = inputs.shape
        flat_input = inputs.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)

        # Compute distances
        embeddings = self.embeddings.weight
        distances = torch.cdist(flat_input.double(), embeddings.double(), p=2)
            
        # Get nearest embedding index
        encoding_indices = torch.argmin(distances, dim=1)  # [B*T]
        
        # Reshape to (B, T)
        return encoding_indices.view(B, T)
        
class OnlineCodebookStocker:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []

    def add(self, z):  # z: (1, T, D)
        if isinstance(z, torch.Tensor):
            z = z.cpu().numpy()
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])  # → (T, D)
        self.vectors.append(z)

        if len(self.vectors) % 100 == 0:
            total = sum([v.shape[0] for v in self.vectors])
            print(f"[Codebook] {len(self.vectors)} utterances, total {total} vectors")

    def get_all(self):
        return np.concatenate(self.vectors, axis=0)
        # return torch.cat(self.vectors, dim=0)
