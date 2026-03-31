import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from train.prepare_data_indv import VCDecBatchCollate_Cycle, VCTKDecDataset_cycle_indv
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

data_dir = 'VCTK_2F2M_train'
val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_vctk.txt'

data_dir_train = 'VCTK_2F2M_train'
data_dir_valid = 'VCTK_2F2M_valid'
enc_dir = 'checkpts/spk_encoder'


import argparse
parser = argparse.ArgumentParser(description='T')
parser.add_argument('--size', type=int, default=1)

argv = parser.parse_args()


epochs = 300
batch_size = 2
learning_rate = 1e-8
codebooksize = argv.size
code_lr = 1e-3
encoder_lr = 1e-4
coef_cyc = 10.0
speakers = ['p236', 'p239', 'p259', 'p263']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    for spk in speakers :
        log_dir = f'log/All_joint_cycle/local/{codebooksize}_diff_{learning_rate}_code_{code_lr}_encoder_{encoder_lr}/{spk}'

        os.makedirs(log_dir, exist_ok=True)
        print(log_dir)
        msg = f"""
        Training ALL Joint
        Codebooksize = {codebooksize}
        Epochs = {epochs}
        batch_size = {batch_size}
        learning rate = {learning_rate}
        spk = {spk}
        code_lr = {code_lr}
        encoder_lr = {encoder_lr}
        coef_cyc = {coef_cyc}
        """
        with open(f'{log_dir}/experiment.log', 'a') as f:
            f.write(msg)

        print('Initializing data loaders...')
        train_set = VCTKDecDataset_cycle_indv(data_dir_train, exc_file, mode = 'train', spk = spk)
        val_set = VCTKDecDataset_cycle_indv(data_dir_valid, exc_file, mode = 'valid', spk = spk)

        collate_fn = VCDecBatchCollate_Cycle()
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
        
        codebook_init = torch.load(f"log/codebook_stock_255_exclude/{spk}_exclude/codebook_stock_{spk}_{codebooksize}.pt")
        quantizer.load_state_dict(codebook_init)

        print('Encoder:')
        print('Number of parameters = %.2fm\n' % (model.encoder.nparams/1e6))
        print('Decoder:')
        print('Number of parameters = %.2fm\n' % (model.decoder.nparams/1e6))

        print('Initializing optimizers...')

        optimizer = torch.optim.Adam([
            {'params': quantizer.parameters(), 'lr': float(code_lr)},
            {'params': model.encoder.parameters(), 'lr': float(encoder_lr)},
            {'params': model.decoder.parameters(), 'lr': float(learning_rate)}
        ])

        # 모델 전체 해제
        for p in model.encoder.parameters():
            p.requires_grad = True
        model.encoder.train()

        # 2. Decoder는 학습 
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
            quantizer.train()
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
                
                
                src_spk = batch['src_spk'][0]
                tgt_spk = batch['tgt_spk'][0]

                # 1. Encoder (Frozen)
                mean_source = model.encoder(mel_source, x_mask) 
                _, vq_loss, _, _, _ = quantizer(mean_source) 

                # ---------------------------------------------------------
                # [Phase 1] A -> B (Fake B 생성)
                # ---------------------------------------------------------
                
                # Map & Quantizer Load
                map_path = f'mappings/{codebooksize}/indv2indv_count/count_matrix_{src_spk}_to_{tgt_spk}.pt'
                count_map_A_to_B = torch.load(map_path, map_location=device)

                quantizer_tgt = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=0.25).to(device)
                quantizer_tgt.load_state_dict(torch.load(f'log/codebook_stock_255_exclude/{tgt_spk}_exclude/codebook_stock_{tgt_spk}_{codebooksize}.pt', map_location=device))

                # Mapping A -> B
                final_map_A_to_B = torch.argmax(count_map_A_to_B, dim=1)
                indices_src = quantizer.get_code_indices(mean_source) 
                indices_tgt = final_map_A_to_B[indices_src.flatten()] 

                target_codewords = quantizer_tgt.embeddings.weight
                final_vectors = target_codewords[indices_tgt] 
                B, D, T = mean_source.shape
                mean_quantized = final_vectors.view(B, T, D).permute(0, 2, 1) # (B, D, T)

                # Ref B (Style)
                mean_ref_source = model.encoder(mel_tgt, tgt_mask) 
                mean_ref_quantized, _, _, _, _ = quantizer_tgt(mean_ref_source)

                ########################################
                indices_src = quantizer.get_code_indices(mean_source)
                src_codewords = quantizer.embeddings.weight
                final_vectors_src = src_codewords[indices_src.flatten()]
                mean_quantized_A = final_vectors_src.view(B, T, D).permute(0, 2, 1)

                # 2. Ref A 준비
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

                # Forward A -> B
                with torch.no_grad():
                    _, mel_prime = model.forward_vq(
                        x = mel_source,           # Guide: A
                        x_lengths = mel_len,
                        mean_quantized = mean_quantized, # Content: B
                        x_ref = mel_tgt,          # Style: B
                        x_ref_lengths = tgt_mel_lengths,
                        mean_ref_quantized = mean_ref_quantized,
                        c = c_tgt,                # Spk: B
                        n_timesteps=6, 
                        mode='ml'
                    )
                
                # ---------------------------------------------------------
                # [Phase 2] B -> A' (Cycle Reconstruction)
                # ---------------------------------------------------------

                # Encode Fake B
                mean_source_prime = model.encoder(mel_prime, x_mask) 
                
                # Map & Quantizer Load
                map_path = f'mappings/{codebooksize}/indv2indv_count/count_matrix_{tgt_spk}_to_{src_spk}.pt'
                count_map_B_to_A = torch.load(map_path, map_location=device)

                quantizer_src = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=0.25).to(device)
                quantizer_src.load_state_dict(torch.load(f'log/codebook_stock_255_exclude/{src_spk}_exclude/codebook_stock_{src_spk}_{codebooksize}.pt', map_location=device))

                # Mapping B -> A
                final_map_B_to_A = torch.argmax(count_map_B_to_A, dim=1)
                
                # [주의] Fake B의 인덱스를 뽑을 때는 Target Quantizer를 써야 정확함
                indices_tgt = quantizer_tgt.get_code_indices(mean_source_prime) 
                indices_src = final_map_B_to_A[indices_tgt.flatten()] 

                src_codewords = quantizer_src.embeddings.weight
                final_vectors_recon = src_codewords[indices_src] 
                
                # [수정 1] view 차원 순서 수정 (B, D, T) -> (B, T, D)
                mean_quantized_A_recon = final_vectors_recon.view(B, T, D).permute(0, 2, 1) 
                
                # Ref A (Style) - 원래 A의 스타일
                mean_ref_quantized_A, _, _, _, _ = quantizer_src(mean_source)
                
                # Forward B -> A'
                mel_double_prime, _ = model.forward_vq(
                    x = mel_prime,            # Guide: Fake B
                    x_lengths = mel_len,
                    mean_quantized = mean_quantized_A_recon, # [수정 2] Content: A (Reconstructed)
                    x_ref = mel_source,       # Style: A
                    x_ref_lengths = mel_len,
                    mean_ref_quantized = mean_ref_quantized_A,
                    c = c,                    # Spk: A
                    n_timesteps=6, 
                    mode='ml'
                )

                # 4. Loss & Backward
                cyc_loss = coef_cyc * F.l1_loss(mel_double_prime, mel_source) / n_mels

                loss = recon_loss + vq_loss + cyc_loss

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
            quantizer.eval()
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
                    
                    src_spk = batch['src_spk'][0]
                    tgt_spk = batch['tgt_spk'][0]

                    # 2. Encoder A & Source Quantization
                    mean_source = model.encoder(mel_source, x_mask)
                    _, vq_loss_val, _, _, _ = quantizer(mean_source)

                    # ---------------------------------------------------------
                    # [Phase 0] Self-Reconstruction (A -> A)
                    # ---------------------------------------------------------
                    # 목적: 모델의 기본 발성 능력 검증 (Recon Loss)
                    
                    indices_src = quantizer.get_code_indices(mean_source)
                    src_codewords = quantizer.embeddings.weight
                    final_vectors_src = src_codewords[indices_src.flatten()]
                    B_size, D, T = mean_source.shape
                    mean_quantized_A = final_vectors_src.view(B_size, T, D).permute(0, 2, 1)

                    mean_ref_quantized_A, _, _, _, _ = quantizer(mean_source)

                    # Forward A -> A
                    _, mel_recon_A = model.forward_vq(
                        x = mel_source,              # Guide: A
                        x_lengths = mel_len,
                        mean_quantized = mean_quantized_A, # Content: A
                        x_ref = mel_source,          # Style: A
                        x_ref_lengths = mel_len,
                        mean_ref_quantized = mean_ref_quantized_A,
                        c = c,                       # Spk: A
                        n_timesteps=6, 
                        mode='ml'
                    )
                    
                    # [Valid Loss 1] Recon Loss
                    recon_loss_val = mse_loss(mel_recon_A, mel_source, x_mask, n_mels)

                    # ---------------------------------------------------------
                    # [Phase 1] A -> B (Fake B Generation)
                    # ---------------------------------------------------------
                    
                    # Load Mappings & Quantizer for Target
                    map_path_A2B = f'mappings/{codebooksize}/indv2indv_count/count_matrix_{src_spk}_to_{tgt_spk}.pt'
                    count_map_A_to_B = torch.load(map_path_A2B, map_location=device)

                    quantizer_tgt = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=0.25).to(device)
                    quantizer_tgt.load_state_dict(torch.load(f'log/codebook_stock_255_exclude/{tgt_spk}_exclude/codebook_stock_{tgt_spk}_{codebooksize}.pt', map_location=device))
                    quantizer_tgt.eval()

                    # Mapping A -> B
                    final_map_A_to_B = torch.argmax(count_map_A_to_B, dim=1)
                    indices_src = quantizer.get_code_indices(mean_source) 
                    indices_tgt = final_map_A_to_B[indices_src.flatten()] 

                    target_codewords = quantizer_tgt.embeddings.weight
                    final_vectors = target_codewords[indices_tgt] 
                    mean_quantized_B = final_vectors.view(B_size, T, D).permute(0, 2, 1) # Content B

                    # Ref B (Style)
                    mean_ref_source = model.encoder(mel_tgt, tgt_mask) 
                    mean_ref_quantized_B, _, _, _, _ = quantizer_tgt(mean_ref_source)

                    # Forward A -> B
                    _, mel_prime = model.forward_vq(
                        x = mel_source,              # Guide: A
                        x_lengths = mel_len,
                        mean_quantized = mean_quantized_B, # Content: B
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
                    # 목적: 변환 후 내용 유지 능력 검증 (Cycle Loss)

                    # Encode Fake B
                    mean_source_prime = model.encoder(mel_prime, x_mask) 
                    
                    # Load Mappings & Quantizer for Source (Back)
                    map_path_B2A = f'mappings/{codebooksize}/indv2indv_count/count_matrix_{tgt_spk}_to_{src_spk}.pt'
                    count_map_B_to_A = torch.load(map_path_B2A, map_location=device)

                    quantizer_src = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=0.25).to(device)
                    quantizer_src.load_state_dict(torch.load(f'log/codebook_stock_255_exclude/{src_spk}_exclude/codebook_stock_{src_spk}_{codebooksize}.pt', map_location=device))
                    quantizer_src.eval()

                    # Mapping B -> A
                    final_map_B_to_A = torch.argmax(count_map_B_to_A, dim=1)
                    
                    # [주의] Fake B의 인덱스 추출 (Target Quantizer)
                    indices_tgt_prime = quantizer_tgt.get_code_indices(mean_source_prime) 
                    indices_src_recon = final_map_B_to_A[indices_tgt_prime.flatten()] 

                    src_codewords = quantizer_src.embeddings.weight
                    final_vectors_recon = src_codewords[indices_src_recon] 
                    mean_quantized_A_recon = final_vectors_recon.view(B_size, T, D).permute(0, 2, 1) # Content A'
                    
                    # Ref A (Style)
                    mean_ref_quantized_A, _, _, _, _ = quantizer_src(mean_source)

                    # Forward B -> A'
                    mel_double_prime, _ = model.forward_vq(
                        x = mel_prime,               # Guide: Fake B
                        x_lengths = mel_len,
                        mean_quantized = mean_quantized_A_recon, # Content: A'
                        x_ref = mel_source,          # Style: A
                        x_ref_lengths = mel_len,
                        mean_ref_quantized = mean_ref_quantized_A,
                        c = c,                       # Spk: A
                        n_timesteps=6, 
                        mode='ml'
                    )

                    # [Valid Loss 2] Cycle Loss
                    cyc_loss_val = coef_cyc * F.l1_loss(mel_double_prime, mel_source) / n_mels

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
                    'quantizer': quantizer.state_dict(),
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
                    'quantizer': quantizer.state_dict(),
                    'model': model.state_dict(),
                    'epoch': epoch + 1
                }
                torch.save(checkpoint, f"{log_dir}/best_model.pt") # 이름도 best_model로 변경 추천
                print(f"  >>> Best Model Saved (Loss: {best_val_loss:.4f})")

        print("All Training Finished.")