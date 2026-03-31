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
#parser.add_argument('--spk', type=str, default='1')

argv = parser.parse_args()


epochs = 100
batch_size = 1
learning_rate = 1e-8
codebooksize = argv.size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    log_dir = f'log/Decoder_cycle_only/global/{codebooksize}_diff_{learning_rate}'

    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    msg = f"""
    Training Decoder Cycle
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
    
    codebook_init = torch.load(f"/home/smin1363/speechst2/VQ-experiment/log/codebook_stock_255_exclude/global/codebook_stock_{codebooksize}.pt")
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
        total_cycle_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            mel_source = batch['mel1'].to(device)
            mel_len = batch['mel_lengths'].to(device) 
            x_mask = sequence_mask(mel_len).unsqueeze(1).to(mel_source.dtype)
            c = batch['c'].to(device)

            mel_tgt = batch['mel_tgt'].cuda()
            tgt_mel_lengths = batch['tgt_mel_lengths'].cuda()    
            tgt_mask = sequence_mask(tgt_mel_lengths).unsqueeze(1).to(mel_tgt.dtype)

            c_tgt = batch['tgt_c'].cuda()
            coef_cyc = 1.0
            
            mean_source = model.encoder(mel_source, x_mask)
            _, vq_loss, _, _, _ = quantizer(mean_source) 

            # [차이점 1] Global Quantizer 사용 (Map 불필요)
            # A의 내용을 담은 코드 (Content A)
            indices_src = quantizer.get_code_indices(mean_source)
            src_codewords = quantizer.embeddings.weight
            final_vectors_src = src_codewords[indices_src.flatten()]
            B, D, T = mean_source.shape

            mean_quantized_A = final_vectors_src.view(B, T, D).permute(0, 2, 1)


            # Ref B (Style)
            mean_ref_source = model.encoder(mel_tgt, tgt_mask) 
            mean_ref_quantized_B, _, _, _, _ = quantizer(mean_ref_source) # 공유 Quantizer

            # ---------------------------------------------------------
            # [Phase 1] A -> B (Fake B 생성)
            # ---------------------------------------------------------
            # [차이점 2] 매핑 없이 mean_quantized_A를 그대로 넣음
            # "내용은 A 그대로(공용어), 목소리만 B로 바꿔라"
            _, mel_prime = model.forward_vq(
                x = mel_source,              # Guide: A
                x_lengths = mel_len,
                mean_quantized = mean_quantized_A, # Content: A (그대로 사용!)
                x_ref = mel_tgt,             # Style: B
                x_ref_lengths = tgt_mel_lengths,
                mean_ref_quantized = mean_ref_quantized_B,
                c = c_tgt,                   # Spk: B
                n_timesteps=6, 
                mode='ml'
            )

            # ---------------------------------------------------------
            # [Phase 2] B -> A' (Cycle Reconstruction)
            # ---------------------------------------------------------
            
            # Encode Fake B
            mean_source_prime = model.encoder(mel_prime, x_mask)
            
            # [차이점 3] 다시 인코딩해서 그대로 사용 (역매핑 불필요)
            # Global Codebook이므로 인코더가 알아서 A와 같은 코드를 찾아줌
            indices_prime = quantizer.get_code_indices(mean_source_prime)
            final_vectors_prime = src_codewords[indices_prime.flatten()]
            mean_quantized_prime = final_vectors_prime.view(B, T, D).permute(0, 2, 1)

            # Ref A (Style)
            mean_ref_quantized_A, _, _, _, _ = quantizer(mean_source)

            # Forward B -> A'
            _, mel_double_prime = model.forward_vq(
                x = mel_prime,               # Guide: Fake B
                x_lengths = mel_len,
                mean_quantized = mean_quantized_prime, # Content: B (재인코딩된 것)
                x_ref = mel_source,          # Style: A
                x_ref_lengths = mel_len,
                mean_ref_quantized = mean_ref_quantized_A,
                c = c,                       # Spk: A
                n_timesteps=6, 
                mode='ml'
            )



            ######
            mean_ref_quantized_A, _, _, _, _ = quantizer(mean_source)

            _, mel_recon_A = model.forward_vq(
                x = mel_source,              # Guide: A
                x_lengths = mel_len,
                mean_quantized = mean_quantized_A, # Content: A (자기 자신)
                x_ref = mel_source,          # Style: A
                x_ref_lengths = mel_len,
                mean_ref_quantized = mean_ref_quantized_A,
                c = c,                       # Spk: A
                n_timesteps=6, 
                mode='ml'
            )

            # 4. Recon Loss 계산
            recon_loss = mse_loss(mel_recon_A, mel_source, x_mask, n_mels)
            cyc_loss = coef_cyc * F.l1_loss(mel_double_prime, mel_source) / n_mels

            ###
            # Loss 계산 (동일)
            loss = recon_loss + cyc_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_cycle_loss += cyc_loss.item()
            pbar.set_postfix({'Recon': recon_loss.item(), 'VQ': vq_loss.item(), 'Cycle': cyc_loss.item()})

        avg_train_recon = total_recon_loss / len(train_loader)
        avg_train_vq = total_vq_loss / len(train_loader)
        avg_train_cycle = total_cycle_loss / len(train_loader)
        # [Validation Loop]
        model.decoder.eval()
        val_recon_loss = 0
        val_vq_loss = 0
        val_cycle_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{epochs} [Valid]"):
                # 1. Data Loading
                mel_source = batch['mel1'].to(device)
                mel_len = batch['mel_lengths'].to(device) 
                x_mask = sequence_mask(mel_len).unsqueeze(1).to(mel_source.dtype)
                c = batch['c'].to(device)

                mel_tgt = batch['mel_tgt'].cuda()
                tgt_mel_lengths = batch['tgt_mel_lengths'].cuda()    
                tgt_mask = sequence_mask(tgt_mel_lengths).unsqueeze(1).to(mel_tgt.dtype)

                c_tgt = batch['tgt_c'].cuda()
                
                mean_source = model.encoder(mel_source, x_mask)
                _, vq_loss, _, _, _ = quantizer(mean_source) 

                # [차이점 1] Global Quantizer 사용 (Map 불필요)
                # A의 내용을 담은 코드 (Content A)
                indices_src = quantizer.get_code_indices(mean_source)
                src_codewords = quantizer.embeddings.weight
                final_vectors_src = src_codewords[indices_src.flatten()]
                mean_quantized_A = final_vectors_src.view(B, T, D).permute(0, 2, 1)

                # Ref B (Style)
                mean_ref_source = model.encoder(mel_tgt, tgt_mask) 
                mean_ref_quantized_B, _, _, _, _ = quantizer(mean_ref_source) # 공유 Quantizer

                # ---------------------------------------------------------
                # [Phase 1] A -> B (Fake B 생성)
                # ---------------------------------------------------------
                # [차이점 2] 매핑 없이 mean_quantized_A를 그대로 넣음
                # "내용은 A 그대로(공용어), 목소리만 B로 바꿔라"
                _, mel_prime = model.forward_vq(
                    x = mel_source,              # Guide: A
                    x_lengths = mel_len,
                    mean_quantized = mean_quantized_A, # Content: A (그대로 사용!)
                    x_ref = mel_tgt,             # Style: B
                    x_ref_lengths = tgt_mel_lengths,
                    mean_ref_quantized = mean_ref_quantized_B,
                    c = c_tgt,                   # Spk: B
                    n_timesteps=6, 
                    mode='ml'
                )

                # ---------------------------------------------------------
                # [Phase 2] B -> A' (Cycle Reconstruction)
                # ---------------------------------------------------------
                
                # Encode Fake B
                mean_source_prime = model.encoder(mel_prime, x_mask)
                
                # [차이점 3] 다시 인코딩해서 그대로 사용 (역매핑 불필요)
                # Global Codebook이므로 인코더가 알아서 A와 같은 코드를 찾아줌
                indices_prime = quantizer.get_code_indices(mean_source_prime)
                final_vectors_prime = src_codewords[indices_prime.flatten()]
                mean_quantized_prime = final_vectors_prime.view(B, T, D).permute(0, 2, 1)

                # Ref A (Style)
                mean_ref_quantized_A, _, _, _, _ = quantizer(mean_source)

                # Forward B -> A'
                _, mel_double_prime = model.forward_vq(
                    x = mel_prime,               # Guide: Fake B
                    x_lengths = mel_len,
                    mean_quantized = mean_quantized_prime, # Content: B (재인코딩된 것)
                    x_ref = mel_source,          # Style: A
                    x_ref_lengths = mel_len,
                    mean_ref_quantized = mean_ref_quantized_A,
                    c = c,                       # Spk: A
                    n_timesteps=6, 
                    mode='ml'
                )
                
                ######
                mean_ref_quantized_A, _, _, _, _ = quantizer(mean_source)

                _, mel_recon_A = model.forward_vq(
                    x = mel_source,              # Guide: A
                    x_lengths = mel_len,
                    mean_quantized = mean_quantized_A, # Content: A (자기 자신)
                    x_ref = mel_source,          # Style: A
                    x_ref_lengths = mel_len,
                    mean_ref_quantized = mean_ref_quantized_A,
                    c = c,                       # Spk: A
                    n_timesteps=6, 
                    mode='ml'
                )

                # 4. Recon Loss 계산
                recon_loss_val = mse_loss(mel_recon_A, mel_source, x_mask, n_mels)

                # [Valid Loss 2] Cycle Loss
                cyc_loss_val = F.l1_loss(mel_double_prime, mel_source) / n_mels

                # Accumulate
                val_recon_loss += recon_loss_val.item()
                val_vq_loss += vq_loss_val.item()
                val_cycle_loss += cyc_loss_val.item()

        # Average calculation
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_vq = val_vq_loss / len(val_loader)
        avg_val_cycle = val_cycle_loss / len(val_loader)
        
        avg_val_total = avg_val_recon + avg_val_vq + avg_val_cycle

        # Logging
        log_msg = (f"Epoch {epoch+1}: "
                    f"Train[R={avg_train_recon:.6f}, VQ={avg_train_vq:.6f}, Cyc={avg_train_cycle:.6f}] | "
                    f"Valid[R={avg_val_recon:.6f}, VQ={avg_val_vq:.6f}, Cyc={avg_val_cycle:.6f}]")
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