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


def inference(vc_model_path, quantizer_path, src_path, tgt_path, output_path, size):

    vc_path = vc_model_path

    generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                    params.layers, params.kernel, params.dropout, params.window_size, 
                    params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                    params.beta_min, params.beta_max)
    if use_gpu:
        generator = generator.cuda()
        generator.load_state_dict(torch.load(vc_path))
    else:
        generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
    generator.eval()

    config_path = "hifi-gan/config.json"
    ckpt_path = "hifi-gan/generator_universal.pth"

    with open(config_path) as f:
        config = json.load(f)
    h = AttrDict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "generator" in ckpt:
        ckpt = ckpt["generator"]

    if use_gpu:
        hifigan_universal = HiFiGAN(h).cuda()
        hifigan_universal.load_state_dict(ckpt)
    else:
        hifigan_universal = HiFiGAN(h)
        hifigan_universal.load_state_dict(ckpt)

    _ = hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()


    from model.codebook import VectorQuantizer

    quantizer = VectorQuantizer(
        embedding_dim=params.embedding_dim,
        num_embeddings=size,
        commitment_cost=params.commitment_cost
    )

    if use_gpu:
        quantizer = quantizer.cuda()
        quantizer.load_state_dict(torch.load(quantizer_path))
    else:
        quantizer.load_state_dict(torch.load(quantizer_path, map_location='cpu'))
    quantizer.eval()


    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') # speaker encoder path
    if use_gpu:
        spk_encoder.load_model(enc_model_fpath, device="cuda")
    else:
        spk_encoder.load_model(enc_model_fpath, device="cpu")

    # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
    src_path = src_path
    tgt_path = tgt_path

    mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
    if use_gpu:
        mel_source = mel_source.cuda()
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
    if use_gpu:
        mel_source_lengths = mel_source_lengths.cuda()

    mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
    if use_gpu:
        mel_target = mel_target.cuda()
    mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
    if use_gpu:
        mel_target_lengths = mel_target_lengths.cuda()

    embed_target = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)
    if use_gpu:
        embed_target = embed_target.cuda()


    # 3. Diffusion decoding (same structure as training)
    mel_encoded, mel_, mean, mean_ref = generator.forward(
        mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target,
        n_timesteps=30, mode='ml'
    )

    mean_quantized, _ ,_ ,_, _= quantizer(mean)

    mean_ref_quantized, _ , _ ,_, _ = quantizer(mean_ref)

    mel_encoded_vq, mel_vq = generator.forward_vq(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target,
        mean_quantized, mean_ref_quantized, n_timesteps=30, mode='ml')


    mel_synth_np = mel_.cpu().detach().squeeze().numpy()
    mel_source_np = mel_.cpu().detach().squeeze().numpy()  
    mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
    if use_gpu:
        mel = mel.cuda()

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
        estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised
