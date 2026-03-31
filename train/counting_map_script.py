# Made by Gemini
import os
import argparse
import numpy as np
from tqdm import tqdm
from itertools import permutations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import params
from model.vc import FwdDiffusion
from model.vc_vq import DiffVC
from model.utils import FastGL, sequence_mask
from model.codebook import VectorQuantizer, OnlineCodebookStocker
from train.prepare_data_indv import VCEncDataset, VCEncBatchCollate, VCTKEncDataset_indv

# =====================================================================
# 1. 터미널 인자 파싱 (경로 유연화)
# =====================================================================
parser = argparse.ArgumentParser(description='Generate VQ Mapping Count Matrix')
parser.add_argument('--size', type=int, default=512, help="Codebook size")
parser.add_argument('--encoder_base_path', type=str, default=None, help="학습된 Encoder 베이스 경로")
parser.add_argument('--codebook_base_path', type=str, default=None, help="학습된 Codebook 베이스 경로")
parser.add_argument('--out_dir', type=str, default=None, help="새로운 맵을 저장할 폴더 경로")
args = parser.parse_args()

codebooksize = args.size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 저장 폴더 설정 (지정 안 하면 기본 경로)
if args.out_dir:
    save_dir = args.out_dir
else:
    save_dir = f'final_mappings/{codebooksize}/indv2indv_count'
os.makedirs(save_dir, exist_ok=True)

# =====================================================================
# 2. 모델 뼈대 초기화
# =====================================================================
print("Initializing Base Model...")
model = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
               params.layers, params.kernel, params.dropout, params.window_size, 
               params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
               params.beta_min, params.beta_max).to(device)

# =====================================================================
# 3. 화자별 조합(Permutations) 루프 시작
# =====================================================================
speakers = ['p236', 'p239', 'p259', 'p263']
data_dir = 'VCTK_2F2M_train'
exc_file = 'filelists/exceptions_vctk.txt'

for (spk1, spk2) in permutations(speakers, 2):
    print(f"\n{'='*50}")
    print(f"Building Map: {spk1} -> {spk2}")
    print(f"{'='*50}")

    # --- 데이터 로더 (Source 화자 spk1 기준) ---
    train_set = VCTKEncDataset_indv(data_dir, exc_file, avg_type='mode', speaker=spk1)
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=None)

    # -----------------------------------------------------------------
    # [A] 스마트 Encoder 로드 (spk1 기준)
    # -----------------------------------------------------------------
    if args.encoder_base_path:
        enc_ckpt_path = os.path.join(args.encoder_base_path, spk1, 'best_model.pt')
        if os.path.exists(enc_ckpt_path):
            ckpt_enc = torch.load(enc_ckpt_path, map_location=device)
            if 'encoder' in ckpt_enc:
                model.encoder.load_state_dict(ckpt_enc['encoder'])
                print(f"  -> Loaded custom Encoder for {spk1}.")
            else:
                print(f"  -> [Warning] 'encoder' key missing. Using pretrained for {spk1}.")
                model.load_state_dict(torch.load("log/log_Gunhee/vc_255.pt", map_location=device))
        else:
            model.load_state_dict(torch.load("log/log_Gunhee/vc_255.pt", map_location=device))
    else:
        # 기본 pretrained 로드
        model.load_state_dict(torch.load("log/log_Gunhee/vc_255.pt", map_location=device))
    
    model.eval()
    encoder = model.encoder.eval()

    # -----------------------------------------------------------------
    # [B] 스마트 Codebook 로드 (spk1 & spk2)
    # -----------------------------------------------------------------
    quantizer_indv_src = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=params.commitment_cost).to(device)
    quantizer_indv_tgt = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=params.commitment_cost).to(device)

    if args.codebook_base_path:
        src_cb_path = os.path.join(args.codebook_base_path, spk1, 'best_model.pt')
        tgt_cb_path = os.path.join(args.codebook_base_path, spk2, 'best_model.pt')
        
        ckpt_cb_src = torch.load(src_cb_path, map_location=device)
        ckpt_cb_tgt = torch.load(tgt_cb_path, map_location=device)
        
        quantizer_indv_src.load_state_dict(ckpt_cb_src['quantizer'] if 'quantizer' in ckpt_cb_src else ckpt_cb_src)
        quantizer_indv_tgt.load_state_dict(ckpt_cb_tgt['quantizer'] if 'quantizer' in ckpt_cb_tgt else ckpt_cb_tgt)
        print(f"  -> Loaded custom Codebooks for {spk1} and {spk2}.")
    else:
        src_cb_path = f"log/codebook_stock_255_exclude/{spk1}_exclude/codebook_stock_{spk1}_{codebooksize}.pt"
        tgt_cb_path = f"log/codebook_stock_255_exclude/{spk2}_exclude/codebook_stock_{spk2}_{codebooksize}.pt"
        
        quantizer_indv_src.load_state_dict(torch.load(src_cb_path, map_location=device))
        quantizer_indv_tgt.load_state_dict(torch.load(tgt_cb_path, map_location=device))
        print("  -> Loaded Stock Codebooks.")

    quantizer_indv_src.eval()
    quantizer_indv_tgt.eval()

    # -----------------------------------------------------------------
    # [C] Counting Map 생성 및 카운트
    # -----------------------------------------------------------------
    count_indv2indv = torch.zeros((codebooksize, codebooksize), dtype=torch.long).to(device)

    print("  -> Starting mapping count...")
    with torch.no_grad():
        for batch in tqdm(train_loader, leave=False):
            mel_source = batch['x'].to(device)
            mel_source_lengths = torch.LongTensor([mel_source.shape[-1]]).to(device)
            x_mask = sequence_mask(mel_source_lengths).unsqueeze(1).to(mel_source.dtype)
            
            # Encoder로 특징 추출 (B, D, T)
            mean = encoder(mel_source, x_mask) 
            
            # 양측 코드북으로 양자화하여 인덱스 추출 (B, T)
            _, _, _, _, indices_src = quantizer_indv_src(mean) 
            _, _, _, _, indices_tgt = quantizer_indv_tgt(mean)

            indices_src = indices_src.flatten()
            indices_tgt = indices_tgt.flatten()

            # 카운트 매트릭스 업데이트
            for i_src, i_tgt in zip(indices_src, indices_tgt):
                count_indv2indv[i_src, i_tgt] += 1

    # -----------------------------------------------------------------
    # [D] 완성된 Map 저장
    # -----------------------------------------------------------------
    save_path = os.path.join(save_dir, f'count_matrix_{spk1}_to_{spk2}.pt')
    torch.save(count_indv2indv, save_path)
    print(f"  => Saved to {save_path}")

print("\nAll mappings successfully generated!")