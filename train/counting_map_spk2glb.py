import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.vc import FwdDiffusion
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from train.prepare_data_indv import VCEncDataset, VCEncBatchCollate, VCTKEncDataset_indv
import params
from model.data_cycle_4speakers import VCDecBatchCollate, VCTKDecDataset
from model.vc_vq import DiffVC
from model.utils import FastGL
from model.codebook import VectorQuantizer, OnlineCodebookStocker # <-- 추가

# model 구성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
embedding_dim = params.embedding_dim
dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max
num_embeddings = params.num_code
random_seed = params.seed
test_size = params.test_size

import argparse

parser = argparse.ArgumentParser(description='T')
parser.add_argument('--codebooksize', type=int,default=10)
argv = parser.parse_args()


codebooksize = argv.codebooksize

# Step 1 : Model 불러오기
print("Loading encoder...")
model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                dec_dim, beta_min, beta_max).cuda()

model.load_state_dict(torch.load("log/log_Gunhee/vc_255.pt"))
model.eval()
encoder = model.encoder.eval()

# Step 2: Load dataset&Loader
print("Loading dataset...")
data_dir = 'VCTK_2F2M_train'
exc_file = 'filelists/exceptions_vctk.txt'
avg_type = 'mode'

from itertools import permutations

speakers = ['p236', 'p239', 'p259', 'p263']
for (spk1, spk2) in permutations(speakers, 2):
    train_set = VCTKEncDataset_indv(data_dir, exc_file, avg_type='mode', speaker=spk1)
    collate_fn = VCEncBatchCollate()
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=None)


    print("Loading indv VQ...")
    quantizer_indv_src = VectorQuantizer( embedding_dim= params.embedding_dim, num_embeddings= codebooksize, 
                                commitment_cost= params.commitment_cost).cuda()
    codebook_init = torch.load(f"log/codebook_stock_255_exclude/{spk1}_exclude/codebook_stock_{spk1}_{codebooksize}.pt")
    
    quantizer_indv_src.load_state_dict(codebook_init)

    quantizer_indv_tgt = VectorQuantizer( embedding_dim= params.embedding_dim, num_embeddings= codebooksize, 
                                commitment_cost= params.commitment_cost).cuda()
    codebook_init = torch.load(f"log/codebook_stock_255_exclude/{spk2}_exclude/codebook_stock_{spk2}_{codebooksize}.pt")
    quantizer_indv_tgt.load_state_dict(codebook_init)

    # count_matrix[i, j] = A의 i번째 인덱스와 Global의 j번째 인덱스가 동시에 뽑힌 횟수
    count_indv2indv = torch.zeros(
        (codebooksize, codebooksize), 
        dtype=torch.long
    ).to(device)

    from model.utils import sequence_mask

    print("Starting mapping count...")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            # 1. z 벡터 추출 (기존 코드와 동일)
            
            mel_source = batch['x'].to(device)
            mel_source_lengths = torch.LongTensor([mel_source.shape[-1]]).to(device)
            x_mask = sequence_mask(mel_source_lengths).unsqueeze(1).to(mel_source.dtype)
            
            mean = encoder(mel_source, x_mask) # (B, D, T)
            
            # VQ 입력 형식 (B, T, D)로 변경
            z = mean
            # vq_A 통과
            _, _, _, _, indices_src = quantizer_indv_src(z) # indices_A shape: (B, T)
            
            # vq_global 통과
            _, _, _, _, indices_tgt = quantizer_indv_tgt(z) # indices_global shape: (B, T)

            # 3. Count 매트릭스 업데이트
            # (B, T) 형태의 인덱스를 1차원으로 펼침
            indices_src = indices_src.flatten()
            indices_tgt = indices_tgt.flatten()

            # 각 (idx_A, idx_global) 쌍을 카운트
            for i_src, i_tgt in zip(indices_src, indices_tgt):
                count_indv2indv[i_src, i_tgt] += 1

        print("Count matrix populated.")
        os.makedirs(f'final_mappings/{codebooksize}/indv2indv_count', exist_ok=True) # 저장할 폴더 생성
        torch.save(count_indv2indv, f'final_mappings/{codebooksize}/indv2indv_count/count_matrix_{spk1}_to_{spk2}.pt')

