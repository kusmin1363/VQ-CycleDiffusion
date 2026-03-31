from pymcd.mcd import Calculate_MCD
import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import pprint

def calculate_mcd_msd(validation_dir, gt_dir, output_dir, mcd_mode, model_name, epoch):
    converted_dirs = os.listdir(validation_dir)
    total_mcd_list = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'{output_dir} 폴더가 존재하지 않아 새로 생성했습니다.')

    total_result = []
    results = []
    excel_data = []

    for converted_dir in converted_dirs:
        src, tar = converted_dir.split('_to_')
        converted_wav_dir = os.path.join(validation_dir, converted_dir)
        converted_wavs = os.listdir(converted_wav_dir)

        tmp_list = []
        pair_results = []
        for converted_wav in converted_wavs:
            src_sen, tar_sen = converted_wav.split('_to_')
            tar_sen = tar_sen.replace('.wav', '')  # 확장자 제거

            if src_sen == '001' or tar_sen == '001':
                continue

            target_dir = os.path.join(gt_dir, tar)
            target_wav = os.path.join(target_dir, f'{tar}_{src_sen}_mic1.wav')
            converted = os.path.join(converted_wav_dir, converted_wav)

            if os.path.exists(converted) and os.path.exists(target_wav):
                mcd_toolbox = Calculate_MCD(MCD_mode=mcd_mode)
                mcd_result = mcd_toolbox.calculate_mcd(target_wav, converted)
                tmp_list.append(mcd_result)
                total_result.append(mcd_result)

                pair_results.append({
                    'source_sentence': src_sen,
                    'target_sentence': tar_sen,
                    'mcd_value': mcd_result
                })

                excel_data.append([converted_dir, src_sen, tar_sen, mcd_result])

        if tmp_list:
            mean_mcd = np.mean(tmp_list)
            std_mcd = np.std(tmp_list)
            print(f'{converted_dir} - 평균 MCD: {mean_mcd}, 표준편차: {std_mcd}')
            total_mcd_list.extend(tmp_list)
            results.append({
                'converted_dir': converted_dir,
                'mean_mcd': mean_mcd,
                'std_mcd': std_mcd,
                'pair_mcd_values': pair_results
            })

    total_mean_mcd = np.mean(total_result)
    total_std_mcd = np.std(total_result)

    output_last_dir = f'{model_name}_{epoch}'
    output_file = os.path.join(output_dir, f"{output_last_dir}.json")

    # 파일을 저장할 디렉터리가 없으면 생성
    output_dir_path = os.path.dirname(output_file)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # 결과를 output_dir 폴더 내에 저장
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'결과가 {output_file}에 저장되었습니다.')
    else:
        print(f'{output_file} 파일이 이미 존재합니다.')

    excel_output_file = os.path.join(output_dir, f"{output_last_dir}.xlsx")
    if not os.path.exists(excel_output_file):
        df = pd.DataFrame(excel_data, columns=['Converted Dir', 'Source Sentence', 'Target Sentence', 'MCD Value'])
        df.to_excel(excel_output_file, index=False)
        print(f'결과가 {excel_output_file}에 엑셀 파일로 저장되었습니다.')


# Example usage
epoch = 270
model_name = f'real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50/real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50_270'
converted_dir = f'converted_all'
mcd_mode = 'dtw'
parser = argparse.ArgumentParser(description='T')
parser.add_argument('--test_dir', type=str, default=f'/home/rtrt505/speechst1/CycleDiffusion/{converted_dir}/{model_name}')
parser.add_argument('--gt_dir', type=str, default='/home/rtrt505/speechst1/CycleDiffusion/VCTK_2F2M/wavs/')
parser.add_argument('--output_dir', type=str, default=f'/home/rtrt505/speechst1/CycleDiffusion/calculate/{model_name}')
argv = parser.parse_args()

calculate_mcd_msd(validation_dir=argv.test_dir, gt_dir=argv.gt_dir, output_dir=argv.output_dir, mcd_mode=mcd_mode, model_name=model_name, epoch=epoch)
