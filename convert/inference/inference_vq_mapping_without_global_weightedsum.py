import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
use_gpu = torch.cuda.is_available()
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

import params
from model.vc_vq import DiffVC
import torch.nn.functional as F

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
import torchaudio

from model.codebook import VectorQuantizer

import torch

def patch_unseen_mapping(count_map, quantizer):
    """
    [맵 수리 패치] Unseen Index를 Nearest Seen Index의 매핑 정보로 대체하는 함수
    count_map: (Vocab_Size_In, Vocab_Size_Out) 형태의 카운트 맵
    quantizer: 입력 측의 quantizer (임베딩 벡터를 가져오기 위함)
    """
    try:
        row_sums = count_map.sum(dim=1)
        seen_mask = row_sums > 0
        unseen_mask = row_sums == 0
        
        if unseen_mask.any():
            # 1. 입력 측 Codebook의 실제 임베딩 벡터 가져오기
            embeds = quantizer.embeddings.weight.data # (Vocab_Size_In, D)
            
            # 2. Seen / Unseen 벡터 분리
            seen_indices = torch.nonzero(seen_mask).squeeze()
            seen_embeds = embeds[seen_mask]
            unseen_embeds = embeds[unseen_mask]
            
            # 3. L2 거리 계산
            dist = torch.cdist(unseen_embeds, seen_embeds)
            
            # 4. 가장 가까운 Seen 인덱스 찾기
            nearest_seen_idx_in_subset = dist.argmin(dim=1)
            nearest_seen_idx_global = seen_indices[nearest_seen_idx_in_subset]
            
            # 5. 매핑 테이블 덮어쓰기
            count_map[unseen_mask] = count_map[nearest_seen_idx_global]
            
    except Exception as e:
        print(f"맵 수리 패치 중 에러 발생: {e}")
        
    return count_map


def inference(vc_model_path, src_wav_path, tgt_wav_path, output_path, codebooksize):

    device = torch.device("cuda" if use_gpu else "cpu")

    # --- 1. 모델 로드 (Generator, HiFiGAN, Speaker Encoder) ---
    generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                       params.layers, params.kernel, params.dropout, params.window_size, 
                       params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                       params.beta_min, params.beta_max).to(device)
    generator.load_state_dict(torch.load(vc_model_path, map_location=device))
    generator.eval()

    config_path = "hifi-gan/config.json"
    ckpt_path = "hifi-gan/generator_universal.pth"
    with open(config_path) as f: config = json.load(f)
    h = AttrDict(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "generator" in ckpt: ckpt = ckpt["generator"]
    hifigan_universal = HiFiGAN(h).to(device)
    hifigan_universal.load_state_dict(ckpt)
    _ = hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()
    
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')
    spk_encoder.load_model(enc_model_fpath, device="cuda" if use_gpu else "cpu")

    # --- 2. 경로에서 화자 ID 추출 및 관련 모델/맵 로드 ---
    src_filename = os.path.basename(src_wav_path)
    src_spk = src_filename.split('_')[0]
    tgt_filename = os.path.basename(tgt_wav_path)
    tgt_spk = tgt_filename.split('_')[0]
    print(f"Converting from '{src_spk}' to '{tgt_spk}'...")

    print(codebooksize)
    # VQ Load
    quantizer_src = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=0.25).to(device)
    quantizer_src.load_state_dict(torch.load(f'log/codebook_stock_255_exclude/{src_spk}_exclude/codebook_stock_{src_spk}_{codebooksize}.pt', map_location=device))
    quantizer_src.eval()

    quantizer_tgt = VectorQuantizer(embedding_dim=params.embedding_dim, num_embeddings=codebooksize, commitment_cost=0.25).to(device)
    quantizer_tgt.load_state_dict(torch.load(f'log/codebook_stock_255_exclude/{tgt_spk}_exclude/codebook_stock_{tgt_spk}_{codebooksize}.pt', map_location=device))
    quantizer_tgt.eval()

    mel_source = torch.from_numpy(get_mel(src_wav_path)).float().unsqueeze(0).to(device)
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]]).to(device)
    mel_target = torch.from_numpy(get_mel(tgt_wav_path)).float().unsqueeze(0).to(device)
    mel_target_lengths = torch.LongTensor([mel_target.shape[-1]]).to(device)
    embed_target = torch.from_numpy(get_embed(tgt_wav_path)).float().unsqueeze(0).to(device)

    # --- 4. 변환 로직 (A -> Global -> B) ---
    _, _, mean, mean_ref = generator.forward(
        mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target,
        n_timesteps=30, mode='ml'
    )
    
    map_path = f'mappings/{codebooksize}/indv2indv_count/count_matrix_{src_spk}_to_{tgt_spk}.pt'

    try:
        count_map_A_to_B = torch.load(map_path, map_location=device)
    except FileNotFoundError:
        print(f"오류: A->B 다이렉트 맵 파일이 없습니다!")
        print(f"경로: {map_path}")
        return
    
    count_map_A_to_B = patch_unseen_mapping(count_map_A_to_B, quantizer_src)

    print("Calculating the final A -> B map (Weightedsum)...")
    row_sums = count_map_A_to_B.sum(dim=1, keepdim=True) + 1e-8
    prob_map_A_to_B = count_map_A_to_B.float() / row_sums

    # Step 1: Source Indices 추출
    indices_src = quantizer_src.get_code_indices(mean) # (B, T)
    flat_indices = indices_src.flatten()

    # Step 2: Source Index에 해당하는 Target 확률 분포 가져오기
    # shape: (B*T, Target_Codebook_Size)
    current_probs = prob_map_A_to_B[flat_indices]

    # Step 3: Weighted Sum (확률 x 타겟 코드북)
    target_codewords = quantizer_tgt.embeddings.weight # (Target_Size, D)
    
    # (B*T, Target_Size) @ (Target_Size, D) -> (B*T, D)
    final_vectors = torch.matmul(current_probs, target_codewords)

    B, D, T = mean.shape
    mean_quantized = final_vectors.view(B, T, D).permute(0, 2, 1)

    # mean_ref는 타겟 VQ로 직접 양자화
    mean_ref_quantized, _, _, _, _ = quantizer_tgt(mean_ref)
    # mean_quantized = mean
   # mean_ref_quantized = mean_ref


    mel_encoded_vq, mel_vq = generator.forward_vq(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target,
        mean_quantized, mean_ref_quantized, n_timesteps=30, mode='ml')
    
    mel_synth_np_vq = mel_vq.cpu().detach().squeeze().numpy()
    mel_source_np_vq = mel_vq.cpu().detach().squeeze().numpy() 
    mel_vq = torch.from_numpy(mel_spectral_subtraction(mel_synth_np_vq, mel_source_np_vq, smoothing_window=1)).float().unsqueeze(0)
    if use_gpu:
        mel_vq = mel_vq.cuda()


    with torch.no_grad():
        audio_vq = hifigan_universal.forward(mel_vq).cpu().squeeze().clamp(-1, 1)
        print(audio_vq.shape)
    sr = 22050
    sf.write(f'{output_path}', audio_vq, sr)
    print(f"Converted audio saved to: {output_path}")



def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i:i+2*w+1])
        y[i] = min(x[i+w+1], med)
    return y

def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    print(mel_len)
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, w = smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised
