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
    converted_dirs = os.listdir(validation_dir) # validation: 변환 음성
    total_mcd_list = []

    # output_dir 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'{output_dir} 폴더가 존재하지 않아 새로 생성했습니다.')

    total_result = []
    results = []
    individual_results = [] # 개별 파일의 MCD 결과를 저장할 리스트 추가

    # [최적화 포인트] 인스턴스 생성은 루프 밖에서 한 번만 수행하여 연산 속도 향상
    mcd_toolbox = Calculate_MCD(MCD_mode=mcd_mode)

    for converted_dir in converted_dirs:
        src, tar = converted_dir.split('_to_')
        converted_wav_dir = os.path.join(validation_dir, converted_dir)
        converted_wavs = os.listdir(converted_wav_dir)

        tmp_list = list()
        for converted_wav in converted_wavs:
            src_sen , tar_sen = converted_wav.split('_to_')
            
            if src_sen == '001' or tar_sen == '001.wav':
                continue
                
            target_dir = os.path.join(gt_dir, tar) 
            target_wav = os.path.join(target_dir, f'{tar}_{src_sen}.wav') # 원본음성
            converted = os.path.join(converted_wav_dir ,converted_wav) # 변환음성
            
            if os.path.exists(converted) and os.path.exists(target_wav):
                # MCD 계산
                mcd_result = mcd_toolbox.calculate_mcd(target_wav, converted)
                
                tmp_list.append(mcd_result)
                total_result.append(mcd_result)
                
                # [추가된 기능] 개별 문장의 결과 기록
                individual_results.append({
                    'directory': converted_dir,
                    'source_sentence': src_sen,
                    'target_sentence': tar_sen.replace('.wav', ''),
                    'converted_wav': converted_wav,
                    'mcd_score': mcd_result
                })
            else:
                continue

        # 평균과 표준편차 계산 (기존 유지)
        if tmp_list:
            mean_mcd = np.mean(tmp_list)
            std_mcd = np.std(tmp_list)
            print(f'{converted_dir} - 평균 MCD: {mean_mcd:.4f}, 표준편차: {std_mcd:.4f}')
            total_mcd_list.extend(tmp_list)
            results.append({
                'converted_dir': converted_dir,
                'mean_mcd': mean_mcd,
                'std_mcd': std_mcd
            })
            
    total_mean_mcd = np.mean(total_result)
    total_std_mcd = np.std(total_result)

    # 저장할 파일명 설정
    output_last_dir = f'{model_name}_{epoch}'
    output_file_summary = os.path.join(output_dir, f"{output_last_dir}_summary.json")
    output_file_individual = os.path.join(output_dir, f"{output_last_dir}_individual.json")
    
    total_results = {
            'converted_dir': 'TOTAL_AVERAGE',
            'mean_mcd': total_mean_mcd,
            'std_mcd': total_std_mcd
    }
    results.append(total_results)

    # 1. 디렉토리별 평균/표준편차 결과 저장 (기존 기능)
    with open(output_file_summary, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f'평균 요약 결과가 {output_file_summary}에 저장되었습니다.')
    
    # 2. 개별 문장 결과 저장 (새로운 기능)
    with open(output_file_individual, 'w', encoding='utf-8') as f:
        json.dump(individual_results, f, indent=4)
    print(f'개별 문장 로그가 {output_file_individual}에 저장되었습니다.')


convert_type = "src_tgt_argmax"

for size in [
    "enc_False_cb_False_dec_True",
    "enc_False_cb_True_dec_False",
    "enc_False_cb_True_dec_True",
    "enc_True_cb_True_dec_False",
    "enc_True_cb_True_dec_True"]:
#size = 512
    converted_dir = f'final_samples/vq_mapping_train/{convert_type}'
    model_name = f'{size}'

    mcd_mode = 'dtw'
    parser = argparse.ArgumentParser(description='T')
    parser.add_argument('--test_dir', type=str,default=f'/home/smin1363/speechst2/VQ-experiment/{converted_dir}/{model_name}')
    parser.add_argument('--gt_dir', type=str, default='/home/smin1363/speechst2/VQ-experiment/VCTK_2F2M/wavs/')
    parser.add_argument('--output_dir', type=str, default=f'/home/smin1363/speechst2/VQ-experiment/final_mcd_indv/{converted_dir}/{model_name}')
    argv = parser.parse_args()

    calculate_mcd_msd(validation_dir=argv.test_dir, gt_dir=argv.gt_dir, output_dir=argv.output_dir, mcd_mode = mcd_mode, model_name=model_name, epoch=10)


