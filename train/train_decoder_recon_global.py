import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from train.prepare_data import VCDecBatchCollate, VCTKDecDataset
from model.vc_vq_grad_on import DiffVC
from model.utils import FastGL
import torch.nn.functional as F
import random
from model.codebook import VectorQuantizer  
import datetime
from model.utils import sequence_mask, mse_loss

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

val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_vctk.txt'

data_dir_train = 'VCTK_2F2M_train'
data_dir_valid = 'VCTK_2F2M_valid'
enc_dir = 'checkpts/spk_encoder'


import argparse
parser = argparse.ArgumentParser(description='T')
parser.add_argument('--size', type=int, default=1)

argv = parser.parse_args()


epochs = 150
batch_size = 4
learning_rate = 1e-8
codebooksize = argv.size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    log_dir = f'log/Decoder_only/local/{codebooksize}_diff_{learning_rate}/global'

    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    msg = f"""
    Training Decoder Only Global
    Codebooksize = {codebooksize}
    Epochs = {epochs}
    batch_size = {batch_size}
    learning rate = {learning_rate}
    """
    with open(f'{log_dir}/experiment.log', 'a') as f:
        f.write(msg)

    print('Initializing data loaders...')
    train_set = VCTKDecDataset(data_dir_train, exc_file, mode = 'train')
    val_set = VCTKDecDataset(data_dir_valid, exc_file, mode = 'valid')

    collate_fn = VCDecBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=4,
                            drop_last=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=4,
                            drop_last=False, shuffle=False)

    print('Initializing and loading models...')
    fgl = FastGL(n_mels, sampling_rate, n_fft, hop_size).cuda()
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                dec_dim, beta_min, beta_max).cuda()

    model.load_state_dict(torch.load("log/log_Gunhee/vc_255.pt"))


    quantizer = VectorQuantizer( embedding_dim= params.embedding_dim, num_embeddings= codebooksize, 
                                commitment_cost= params.commitment_cost).cuda()
    
    codebook_init = torch.load(f"log/codebook_stock_255_exclude/global/codebook_stock_{codebooksize}.pt")
    quantizer.load_state_dict(codebook_init)
    quantizer.eval()

    print('Encoder:')
    print('Number of parameters = %.2fm\n' % (model.encoder.nparams/1e6))
    print('Decoder:')
    print('Number of parameters = %.2fm\n' % (model.decoder.nparams/1e6))

    print('Initializing optimizers...')

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': float(learning_rate)}
    ])

    # encoder 고정
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    # decoder 학습 
    for p in model.decoder.parameters():
        p.requires_grad = True 
    model.decoder.train()

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0

    # 학습 방식
    print(f"Start Training... Epochs: {epochs}, Batch: {batch_size}")

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.decoder.train()
        total_recon_loss = 0
        total_vq_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            mel_source = batch['mel1'].to(device)
            mel_len = batch['mel_lengths'].to(device) 
            x_mask = sequence_mask(mel_len).unsqueeze(1).to(mel_source.dtype)
            c = batch['c'].to(device)

            # 1. Encoder (Frozen)
            with torch.no_grad():
                z = model.encoder(mel_source, x_mask) 

            with torch.no_grad():
                z_q, vq_loss, _, _, _ = quantizer(z) 

            # 3. Decoder (Trainable)
            _, mel_pred = model.forward_vq(
                x = z_q,                # Input (Condition): Quantized Latent
                x_lengths = mel_len,             # Mask
                mean_quantized=z_q,           # mean: 보통 Encoder Output을 의미 (여기선 Quantized된 값 사용)
                x_ref=mel_source,     # ref: 스타일 참고용 (복원이므로 자기 자신 mel)
                x_ref_lengths=mel_len,    # ref_mask: 자기 자신 mask
                mean_ref_quantized=z_q,         # mean_ref: 참고 오디오의 Latent (Quantization 전의 z 사용)
                c=c,                # c: 화자 임베딩
                n_timesteps=6, 
                mode='ml'
                )


            # 4. Loss & Backward
            recon_loss = mse_loss(mel_pred, mel_source, x_mask, n_mels)
            loss = recon_loss + vq_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            pbar.set_postfix({'Recon': recon_loss.item(), 'VQ': vq_loss.item()})

        avg_train_recon = total_recon_loss / len(train_loader)
        avg_train_vq = total_vq_loss / len(train_loader)

        model.decoder.eval()
        val_recon_loss = 0
        val_vq_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{epochs} [Valid]"):
                mel_source = batch['mel1'].to(device)
                mel_len = batch['mel_lengths'].to(device) 
                x_mask = sequence_mask(mel_len, mel_source.size(2)).unsqueeze(1).to(mel_source.dtype)
                c = batch['c'].to(device)

                z = model.encoder(mel_source, x_mask)
                z_q, vq_loss, _, _, _ = quantizer(z)

                _, mel_pred = model.forward_vq(
                    x = z_q,                # Input (Condition): Quantized Latent
                    x_lengths = mel_len,             # Mask
                    mean_quantized=z_q,           # mean: 보통 Encoder Output을 의미 (여기선 Quantized된 값 사용)
                    x_ref=mel_source,     # ref: 스타일 참고용 (복원이므로 자기 자신 mel)
                    x_ref_lengths=mel_len,    # ref_mask: 자기 자신 mask
                    mean_ref_quantized=z_q,         # mean_ref: 참고 오디오의 Latent (Quantization 전의 z 사용)
                    c=c,                # c: 화자 임베딩
                    n_timesteps=6, 
                    mode='ml'
                    )                
                recon_loss = mse_loss(mel_pred, mel_source, x_mask, n_mels)

                val_recon_loss += recon_loss.item()
                val_vq_loss += vq_loss.item()

        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_vq = val_vq_loss / len(val_loader)
        avg_val_total = avg_val_recon + avg_val_vq


        log_msg = (f"Epoch {epoch+1}: "
                    f"Train[R={avg_train_recon:.6f}, VQ={avg_train_vq:.6f}] | "
                    f"Valid[R={avg_val_recon:.6f}, VQ={avg_val_vq:.6f}]")
        print(log_msg)
        
        with open(f'{log_dir}/train_log.txt', 'a') as f:
            f.write(log_msg + "\n")

        # Save Checkpoint (Periodic)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict() # 나중에 이어서 학습할 때 필요
            }
            torch.save(checkpoint, f"{log_dir}/checkpoint_ep{epoch+1}.pt")
            print(f"Saved Checkpoint to {log_dir}/checkpoint_ep{epoch+1}.pt")
        
        # Save Best Model
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(checkpoint, f"{log_dir}/best_model.pt") # 이름도 best_model로 변경 추천
            print(f"  >>> Best Model Saved (Loss: {best_val_loss:.4f})")

    print("All Training Finished.")