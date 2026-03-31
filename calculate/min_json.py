import os
import json

# 특정 폴더 경로 설정
folder_path = './real_last_cycle_train_dec_4speakers_original'

# last_train_dec_4speakers_original
# last_train_dec_4speakers_cycle6_lr_3e5_from0
# last_train_dec_4speakers_cycle6_lr_3e5_from50
# last_train_dec_4speakers_cycle6_lr_3e5_from100

# real_last_cycle_train_dec_4speakers_all_cycle6_from_50
# real_last_cycle_train_dec_4speakers_cycle6_from_50
# real_last_cycle_train_dec_4speakers_original
# real_last_cycle_train_dec_4speakers_original_one_hot

# real_last_cycle_train_dec_4speakers_cycle6_from_0
# real_last_cycle_train_dec_4speakers_all_cycle6_from_0

# real_last_cycle_train_dec_4speakers_cycle6_from_100
# real_last_cycle_train_dec_4speakers_all_cycle6_from_100

# real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50

# 변수 초기화
min_mcd = float('inf')
min_file = None

# 폴더 내 모든 파일을 순회
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # JSON 파일 열기
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # 마지막 데이터의 mean_mcd 가져오기
            last_mean_mcd = data[-1]['mean_mcd']
            last_std_mcd = data[-1]['std_mcd']
            # 가장 작은 mean_mcd 값과 파일 기록
            if last_mean_mcd < min_mcd:
                min_mcd = last_mean_mcd
                min_file = filename

# 결과 출력
if min_file:
    print(f"가장 작은 mean_mcd 값을 가진 파일: {min_file}, mean_mcd: {min_mcd}")
else:
    print("해당 폴더에 JSON 파일이 없습니다.")