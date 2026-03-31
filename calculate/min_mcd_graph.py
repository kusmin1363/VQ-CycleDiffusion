import os
import json

# 폴더 경로 설정
folder_path = './'

# 결과를 저장할 리스트
mean_mcd_values = []
model_name = 'last_train_dec_2speakers_cycle6'
# last_train_dec_2speakers_cycle6
# last_train_dec_2speakers_cycle8
# last_train_dec_2speakers_cycle_linear
seenORunseen = 'seen'
start = 5
finish = 301
step = 5
# 200부터 400까지 5단위로 숫자 생성
for number in range(start, finish+1, step):
    number = str(number)
    filename = f'mcd_{seenORunseen}_{model_name}_{number}.json'
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):  # 파일이 실제로 존재하는지 확인
        with open(file_path, 'r') as file:
            data = json.load(file)
            mean_mcd_values.append(float(data['mean_mcd']))  # mean_mcd 값 리스트에 추가


# 리스트에서 최소값 찾기
min_value = min(mean_mcd_values)

# 최소값의 인덱스 찾기
min_index = mean_mcd_values.index(min_value)*10+200

# 결과 출력
print(f"Minimum Value: {min_value}, Index: {min_index}")


import matplotlib.pyplot as plt

# 200부터 400까지 5단위로 숫자 생성하여 epoch 리스트 생성
epochs = list(range(start, finish+1, step))

# mean_mcd_values 리스트가 이미 채워져 있다고 가정

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_mcd_values, marker='o', linestyle='-', color='blue')
plt.title('Mean MCD Values over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean MCD Value')
plt.grid(True)

# 그래프를 이미지 파일로 저장
plt.savefig(f'mean_mcd_values_over_epochs_{seenORunseen}_{model_name}.png')

# 파일 저장 확인 메시지
print(f"그래프가 'mean_mcd_values_over_epochs.png_{seenORunseen}_{model_name}'로 저장되었습니다.")
