import os, sys
import model_4style
import argparse
import soundfile
import torch
import numpy as np
import pickle
import json

from speech_tools import world_decode_mc, world_speech_synthesis
import data_manager as dm


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_normalized_t = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_normalized_t


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='')
parser.add_argument('--convert_path', default='')
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(args.seed)
spk_list = ["p238", "p261", "p334", "p360"]

TOTAL_SPK_NUM = len(spk_list)
print("TOTAL SPK NUM >> ",TOTAL_SPK_NUM)

SPK_DICT = {
    spk_idx:spk_id 
    for spk_idx, spk_id in enumerate(spk_list)
}
VEC_DICT = {
    spk_id:[make_one_hot_vector(spk_idx, len(spk_list))]
    for spk_idx, spk_id in SPK_DICT.items()
}


model_dir = args.model_path
convert_path = args.convert_path

print("-----------")

VAE = model_4style.VAE()
vc_path = vc_model_path

generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                params.layers, params.kernel, params.dropout, params.window_size, 
                params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                params.beta_min, params.beta_max)
generator = generator.cuda()
generator.load_state_dict(torch.load(vc_path))
generator.eval()


if args.epoch == 0 :
    VAE.load_state_dict(torch.load(model_dir+"/final_{}.pt"))
else:
    VAE.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_{}.pt"))




feat_dir = os.path.join("/nfs/speechst2/storage/klklp98/Exp_Disentanglement/data_vcc_20230304")

sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128

STAT_DICT = dict()
for source_spk in spk_list:
    stat_path = "/nfs/speechst2/storage/klklp98/Exp_Disentanglement/data_vcc_20230304/holeset/train/"+source_spk+"/feats.p"
    _, sp_m, sp_s, logf0_m, logf0_s, _ = load_pickle(stat_path)
    STAT_DICT[source_spk] = (sp_m, sp_s, logf0_m, logf0_s)
    for target_spk in spk_list:
        os.makedirs(os.path.join(convert_path, source_spk+"_to_"+target_spk), exist_ok=True)


for (src_idx, source_spk) in SPK_DICT.items():
    print("Processing", source_spk)
    feat_path = os.path.join(feat_dir,source_spk)    
    sp_m_s, sp_s_s, logf0_m_s, logf0_s_s = STAT_DICT[source_spk]

    # one hot src
    # SPK_DICT[source_spk]
    one_hot_x = dm.make_spk_vector([src_idx], TOTAL_SPK_NUM, 1, is_MD=False)

    for _, _, file_list in os.walk(feat_path):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36" or utt_id=='feats':
                continue
            print("\tConvert {}.wav ...".format(utt_id))

            
            file_path = os.path.join(feat_path, file_id)
            sp, src, f0, ap, _ = load_pickle(file_path)
            #src, f0, ap = load_pickle(file_path)
            

            # src = (coded_sp-sp_m_s) / sp_s_s
            src = np.expand_dims(src, axis=0)
            src = np.expand_dims(src, axis=0)
            src = torch.Tensor(src).float().cuda().contiguous()
            
            # logf0_norm = (np.log(f0)-logf0_m_s) / logf0_s_s

            for (tar_idx, target_spk) in SPK_DICT.items():
                # one_hot_y = VEC_DICT[target_spk]
                one_hot_y = dm.make_spk_vector([tar_idx], TOTAL_SPK_NUM, 1)

                with torch.no_grad():
                    _, _,  _, y_prime_mu, _, y_prime = VAE(x=src, one_hot_src=one_hot_x , one_hot_tar=one_hot_y)

                sp_m_t, sp_s_t, logf0_m_t, logf0_s_t = STAT_DICT[target_spk]
                # mc_t = y_prime_mu.double().cpu().numpy()[0][0]

                # mean as sample
                mc_t_mean = y_prime_mu.double().cpu().numpy()[0][0]
                mc_t_mean = mc_t_mean * sp_s_t + sp_m_t
                mc_t_mean = mc_t_mean.T
                mc_t_mean = np.ascontiguousarray(mc_t_mean)
                sp_t_mean = world_decode_mc(mc = mc_t_mean, fs = sampling_rate)
                new_sp_mean = sp_t_mean

                
                # # 여기!!!!
                new_f0 = pitch_conversion(f0 = f0, mean_log_src = logf0_m_s, std_log_src = logf0_s_s, mean_log_target = logf0_m_t, std_log_target = logf0_s_t)

                # mean as sample
                wav_transformed_mean = world_speech_synthesis(f0=new_f0, decoded_sp=new_sp_mean, 
                    ap=ap, fs=sampling_rate, frame_period=frame_period)
                wav_transformed_mean = np.nan_to_num(wav_transformed_mean)
                soundfile.write(os.path.join(convert_path,source_spk+"_to_"+target_spk, utt_id+".wav"), wav_transformed_mean, sampling_rate)

            
            # print(coded_sp.shape)
            # print(f0.shape)
            # print(ap.shape)