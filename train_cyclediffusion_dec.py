# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied1 warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.



#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

####################################################################################
# Final CycleDiffusion Train Code  2025.04.02
####################################################################################


import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import params
from train.prepare_data import VCDecBatchCollate, VCTKDecDataset
from model.vc_vq_grad_on import DiffVC
from model.utils import FastGL
import torch.nn.functional as F
import random
import json

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
enc_dim = params.enc_dim
dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max
random_seed = params.seed
test_size = params.test_size


data_dir = 'VCTK_2F2M'
val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_vctk.txt'

data_dir_train = 'VCTK_2F2M_train'
data_dir_valid = 'VCTK_2F2M_valid'
enc_dir = 'checkpts/spk_encoder'

epochs = 350
batch_size = 4
learning_rate = 3e-5
log_dir = f'log/log_Gunhee_70'
use_gpu = torch.cuda.is_available()
date = '260102'
if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # random.seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    
    os.makedirs(log_dir, exist_ok=True)
    print('Used Hyperparameters...')
    print(f'Epochs : {epochs}, Learning_rate : {learning_rate}, batch_size : {batch_size}')


    print('Initializing data loaders...')

    print('Initializing data loaders...')
    train_set = VCTKDecDataset(data_dir_train, exc_file, mode = 'train')
    val_set = VCTKDecDataset(data_dir_valid, exc_file, mode = 'valid')

    collate_fn = VCDecBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=0,
                            drop_last=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=0,
                            drop_last=False, shuffle=False)

    print('Initializing and loading models...')
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                   dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                   dec_dim, beta_min, beta_max).cuda()
    model.load_encoder(os.path.join(enc_dir, 'enc.pt'))

    print('Encoder:')
    #print(model.encoder)
    print('Number of parameters = %.2fm\n' % (model.encoder.nparams/1e6))
    print('Decoder:')
    #print(model.decoder)
    print('Number of parameters = %.2fm\n' % (model.decoder.nparams/1e6))

    print('Initializing optimizers...')

    optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True

    iteration = 0
    epoch_continue = 0
    for epoch in range(epoch_continue + 1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        cyc_losses = []
        total_losses = []
        val_losses =[]
        val_cyc_losses = []
        for batch in tqdm(train_loader, total=len(train_set)//batch_size):
            
            model.zero_grad()

            mel, mel_ref = batch['mel1'].cuda(), batch['mel2'].cuda()
            c, mel_lengths = batch['c'].cuda(), batch['mel_lengths'].cuda()
            
            loss = model.compute_loss(mel, mel_lengths, mel_ref, c)
            
            mel_tgt = batch['mel_tgt'].cuda()
            tgt_mel_lengths = batch['tgt_mel_lengths'].cuda()
            c_tgt = batch['tgt_c'].cuda()
            cyc_loss = 0
            if epoch >= 71:
                coef_cyc = 1.0
            else:
                coef_cyc = 0.0
            
            diffusion_step = 6
            iii = 3

            if epoch < 71:
            # with torch.no_grad():
            #     _, mel_prime = model(mel[:iii], mel_lengths[:iii], mel_tgt[:iii], tgt_mel_lengths[:iii], c_tgt[:iii], n_timesteps=diffusion_step, mode='ml')
            # _, mel_double_prime = model(mel_prime, tgt_mel_lengths[:iii], mel[:iii], mel_lengths[:iii], c[:iii], n_timesteps=diffusion_step, mode='ml')         
            # cyc_loss = coef_cyc * F.l1_loss(mel_double_prime, mel[:iii]) / n_mels
                cyc_loss = 0
            else:
                with torch.no_grad():
                    _, mel_prime = model(mel[:iii], mel_lengths[:iii], mel_tgt[:iii], tgt_mel_lengths[:iii], c_tgt[:iii], n_timesteps=diffusion_step, mode='ml')
                _, mel_double_prime = model(mel_prime, tgt_mel_lengths[:iii], mel[:iii], mel_lengths[:iii], c[:iii], n_timesteps=diffusion_step, mode='ml')         
                cyc_loss = coef_cyc * F.l1_loss(mel_double_prime, mel[:iii]) / n_mels

            total_loss = loss + cyc_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
            optimizer.step()
            
            losses.append(loss.item())
            if isinstance(cyc_loss, torch.Tensor):
                cyc_losses.append(cyc_loss.item())
            else:
                cyc_losses.append(cyc_loss)
            total_losses.append(total_loss.item())
            iteration += 1
            print()
            print(f"Iteration: {iteration} / loss: {loss}, cyc_loss: {cyc_loss}, total_loss:{total_loss}")

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader)):

                mel, mel_ref = batch['mel1'].cuda(), batch['mel2'].cuda()
                c, mel_lengths = batch['c'].cuda(), batch['mel_lengths'].cuda()
                
                loss = model.compute_loss(mel, mel_lengths, mel_ref, c)
                
                mel_tgt = batch['mel_tgt'].cuda()
                tgt_mel_lengths = batch['tgt_mel_lengths'].cuda()
                c_tgt = batch['tgt_c'].cuda()
                cyc_loss = 0
                coef_cyc = coef_cyc
                diffusion_step = diffusion_step
                iii = 3

                if epoch >= 71:
                    _, mel_prime = model(mel[:iii], mel_lengths[:iii], mel_tgt[:iii], tgt_mel_lengths[:iii], c_tgt[:iii], n_timesteps=diffusion_step, mode='ml')
                    _, mel_double_prime = model(mel_prime, tgt_mel_lengths[:iii], mel[:iii], mel_lengths[:iii], c[:iii], n_timesteps=diffusion_step, mode='ml')         
                    cyc_loss = coef_cyc * F.l1_loss(mel_double_prime, mel[:iii]) / n_mels
                else:
                    cyc_loss = 0
                
                val_losses.append(loss.item())
                if isinstance(cyc_loss, torch.Tensor):
                    val_cyc_losses.append(cyc_loss.item())
                else:
                    val_cyc_losses.append(cyc_loss)

        train_loss_mean = float(np.mean(losses))
        val_loss_mean   = float(np.mean(val_losses))

        train_cycle_loss_mean = float(np.mean(cyc_losses))
        val_cycle_loss_mean   = float(np.mean(val_cyc_losses))

        total_loss_mean = float(np.mean(total_losses))

        msg = f'Epoch {epoch}: train_loss = {train_loss_mean:.6f}, cyc_loss = {train_cycle_loss_mean:.6f}, total_loss = {total_loss_mean:.6f}, val_loss = {val_loss_mean:.6f}, val_cyc_loss = {val_cycle_loss_mean:.6f}\n'
        print(msg)
        with open(f'{log_dir}/train_cycle.log', 'a') as f:
            f.write(msg)
        
        if epoch % 5 == 0:
            model.eval()
            print('Inference...\n')
            with torch.no_grad():
                print('Saving model...\n')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'loss': loss,
                }, f"{log_dir}/vc_{epoch}_{date}_full.pt")
