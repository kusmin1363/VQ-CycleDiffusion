# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from train.prepare_data import VCTKEncDataset, VCEncBatchCollate
from model.vc import FwdDiffusion
from model.utils import FastGL, sequence_mask

n_mels = params.n_mels
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size
channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
dim = params.enc_dim
random_seed = params.seed
test_size = params.test_size

data_dir = 'VCTK_2F2M'
val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_vctk.txt'
avg_type = 'mode'

use_gpu = torch.cuda.is_available()
log_dir = 'log/log_speaker_encoder'
epochs = 500
batch_size = 32
learning_rate = 5e-4
save_every = 20


if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(log_dir, exist_ok=True)

    print('Initializing data loaders...')

    full_set = VCTKEncDataset(data_dir, exc_file, avg_type)    
    num_total = len(full_set)
    val_ratio = 0.15  # 1763 * 0.15 ~ 260개
    num_val = int(num_total * val_ratio)

    g = torch.Generator()
    g.manual_seed(random_seed)
    indices = torch.randperm(num_total, generator=g).tolist()

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    from torch.utils.data import Subset
    train_set = Subset(full_set, train_indices)
    val_set   = Subset(full_set, val_indices)

    collate_fn = VCEncBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=4,
                            drop_last=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=4,
                            drop_last=False, shuffle=False)

    print('Initializing models...')
    model = FwdDiffusion(n_mels, channels, filters, heads, layers, kernel, 
                         dropout, window_size, dim).cuda()

    print('Encoder:')
    print(model)
    print('Number of parameters = %.2fm\n' % (model.nparams/1e6))

    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, total=len(train_loader)):
            mel_x, mel_y = batch['x'].cuda(), batch['y'].cuda()
            mel_lengths = batch['lengths'].cuda()
            mel_mask = sequence_mask(mel_lengths).unsqueeze(1).to(mel_x.dtype)

            model.zero_grad()
            loss = model.compute_loss(mel_x, mel_y, mel_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            train_losses.append(loss.item())
            iteration += 1

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                mel_x, mel_y = batch['x'].cuda(), batch['y'].cuda()
                mel_lengths = batch['lengths'].cuda()
                mel_mask = sequence_mask(mel_lengths).unsqueeze(1).to(mel_x.dtype)

                loss = model.compute_loss(mel_x, mel_y, mel_mask)
                val_losses.append(loss.item())

        train_loss_mean = float(np.mean(train_losses))
        val_loss_mean   = float(np.mean(val_losses))

        msg = f'Epoch {epoch}: train_loss = {train_loss_mean:.6f}, val_loss = {val_loss_mean:.6f}\n'
        print(msg)
        with open(f'{log_dir}/train_enc.log', 'a') as f:
            f.write(msg)

        if epoch % save_every > 0:
            continue

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/enc_{epoch}.pt")
