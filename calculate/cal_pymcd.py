from pymcd.mcd import Calculate_MCD
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import pprint

# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
def calculate_mcd_msd(validation_dir, gt_dir, output_dir, mcd_mode, model_name, epoch):
    converted_dirs = os.listdir(validation_dir) #validation: 변환 음성
    total_mcd_list = []

    # output_dir 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'{output_dir} 폴더가 존재하지 않아 새로 생성했습니다.')

    total_result = []
    results = []

    for converted_dir in converted_dirs:
        src, tar = converted_dir.split('_to_')
        converted_wav_dir = os.path.join(validation_dir, converted_dir)
        converted_wavs = os.listdir(converted_wav_dir)

        tmp_list = list()
        for converted_wav in converted_wavs:
            src_sen , tar_sen = converted_wav.split('_to_')
            #print(src_sen , tar_sen)
            if src_sen == '001' or tar_sen == '001.wav':
                continue
            target_dir = os.path.join(gt_dir, tar) 
            target_wav = os.path.join(target_dir, f'{tar}_{src_sen}.wav') # 원본음성
            mcd_toolbox = Calculate_MCD(MCD_mode=mcd_mode)
            converted = os.path.join(converted_wav_dir ,converted_wav) # 변환음성
            
            if os.path.exists(converted) and os.path.exists(target_wav):
                mcd_result = mcd_toolbox.calculate_mcd(target_wav, converted)
                tmp_list.append(mcd_result)
                total_result.append(mcd_result)
            else:
                continue

        # 평균과 표준편차 계산
        if tmp_list:
            mean_mcd = np.mean(tmp_list)
            std_mcd = np.std(tmp_list)
            print(f'{converted_dir} - 평균 MCD: {mean_mcd}, 표준편차: {std_mcd}')
            total_mcd_list.extend(tmp_list)
            results.append({
                'converted_dir': converted_dir,
                'mean_mcd': mean_mcd,
                'std_mcd': std_mcd
            })
            #print(results)
    total_mean_mcd = np.mean(total_result)
    total_std_mcd = np.std(total_result)

    # 결과를 output_dir 폴더 내에 저장
    output_last_dir = f'{model_name}_{epoch}'
    output_file = os.path.join(output_dir, f"{output_last_dir}.json")
    print(results)
    
    total_results = {
            'converted_dir': output_last_dir,
            'mean_mcd': total_mean_mcd,
            'std_mcd': total_std_mcd
    }

    results.append(total_results)

    # 결과 저장
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f'결과가 {output_file}에 저장되었습니다.')



# converted_dir = f'samples'
# size = 64
# "samples/vc_255_exclude_64"
# mcd_mode = 'dtw'
# model_name = "vc_255_exclude_16384"
# epoch = 1
# parser = argparse.ArgumentParser(description='T')
# parser.add_argument('--test_dir', type=str,default=f'/home/smin1363/speechst2/VQ-experiment/{converted_dir}/{model_name}')
# parser.add_argument('--gt_dir', type=str, default='/home/smin1363/speechst2/VQ-experiment/VCTK_2F2M/wavs/')
# #parser.add_argument('--summary', type=str, default=f'/home/rtrt505/speechst1/CycleDiffusion/calculate/{model_name}/{source}_to_{target}/mcd_{output_last_dir}.json')
# parser.add_argument('--output_dir', type=str, default=f'/home/smin1363/speechst2/VQ-experiment/mcd/{converted_dir}/{model_name}')
# argv = parser.parse_args()


# calculate_mcd_msd(validation_dir=argv.test_dir, gt_dir=argv.gt_dir, output_dir=argv.output_dir, mcd_mode = mcd_mode, model_name=model_name, epoch=epoch)

convert_type = "global_tgt_weightedsum"

for size in [512, 1024, 2048, 4096, 8192]:
#size = 512
    converted_dir = f'final_samples_fixed/vq_mapping/{convert_type}'
    model_name = f'vq_{size}'

    mcd_mode = 'dtw'
    parser = argparse.ArgumentParser(description='T')
    parser.add_argument('--test_dir', type=str,default=f'/home/smin1363/speechst2/VQ-experiment/{converted_dir}/{model_name}')
    parser.add_argument('--gt_dir', type=str, default='/home/smin1363/speechst2/VQ-experiment/VCTK_2F2M/wavs/')
    parser.add_argument('--output_dir', type=str, default=f'/home/smin1363/speechst2/VQ-experiment/final_mcd/{converted_dir}/{model_name}')
    argv = parser.parse_args()

    calculate_mcd_msd(validation_dir=argv.test_dir, gt_dir=argv.gt_dir, output_dir=argv.output_dir, mcd_mode = mcd_mode, model_name=model_name, epoch=10)


