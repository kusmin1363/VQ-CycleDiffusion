from pymcd.mcd import Calculate_MCD
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import pprint

# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics

def calculate_mcd_msd(validation_dir, gt_dir, summary_json, mcd_mode, model_name):
    converted_dirs = os.listdir(validation_dir) #validation: 변환 음성
    total_mcd_list = []

    with open(summary_json, "r") as j :
        summary = json.load(j)

    for converted_dir in converted_dirs:
        src, tar = converted_dir.split('_to_')
        converted_wav_dir = os.path.join(validation_dir, converted_dir)
        converted_wavs = os.listdir(converted_wav_dir)

        tmp_list = list()
        for converted_wav in converted_wavs:
            src_sen , tar_sen = converted_wav.split('_to_')
            target_dir = os.path.join(gt_dir, tar) 
            target_wav = os.path.join(target_dir, f'{tar}_{src_sen}_mic1.wav') # 원본음성
            mcd_toolbox = Calculate_MCD(MCD_mode=mcd_mode)
            converted = os.path.join(converted_wav_dir ,converted_wav) # 변환음성
            
            if os.path.exists(converted) and os.path.exists(target_wav):
                tmp_list.append(mcd_toolbox.calculate_mcd(target_wav, converted))
                _, c = converted.split(f'/{model_name}/')
                print(f'{tar}_{src_sen}_mic1.wav and {c} => mcd: {tmp_list[-1]}')
            else:
                continue
            
            
        total_mcd_list += tmp_list


    print()
    print('-' * 29 + "TOTAL" + '-' * 29)
    print(' MCD_MEAN: ', np.mean(total_mcd_list), ' MCD_STD: ', np.std(total_mcd_list))
    print()

    summary["mean_mcd"] = str(np.mean(total_mcd_list))
    summary["std_mcd"] = str(np.std(total_mcd_list))
    
    with open(summary_json, "w", encoding="utf-8") as j :
        json.dump(summary, j, ensure_ascii=False, indent='\t')


#######################################################################################################

model_name = 'vc_vctk_cycle_wodyn'
source = 'p239'
target = 'p263'
converted_dir = f'converted_audio/converted_{source}_to_{target}'




output_last_dir = f'{model_name}_target_{target}'


mcd_mode = 'dtw'
parser = argparse.ArgumentParser(description='T')
parser.add_argument('--test_dir', type=str,default=f'/home/rtrt505/speechst1/CycleDiffusion/{converted_dir}/{model_name}/{output_last_dir}')
parser.add_argument('--gt_dir', type=str, default='/home/rtrt505/speechst1/CycleDiffusion/VCTK_2speakers/wavs/')
parser.add_argument('--summary', type=str, default=f'/home/rtrt505/speechst1/CycleDiffusion/calculate/{source}_to_{target}/{model_name}/mcd_{output_last_dir}.json')
argv = parser.parse_args()


calculate_mcd_msd(validation_dir=argv.test_dir, gt_dir=argv.gt_dir, summary_json=argv.summary, mcd_mode = mcd_mode, model_name=model_name)