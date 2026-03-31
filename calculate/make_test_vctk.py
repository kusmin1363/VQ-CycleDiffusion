import os
import shutil

unseen_speakers = ['p252', 'p261', 'p241', 'p238', 'p243', 'p294', 'p334', 'p343', 'p360', 'p362']
unseen_sentences = ['001', '002', '003', '004', '005']

source_directory = '/home/rtrt505/speechst1/DiffVC/VCTK/wavs'
destination_directory = '/home/rtrt505/speechst1/DiffVC/VCTK/test'

# 확인된 화자 및 문장에 해당하는 파일을 대상 디렉토리로 복사
for speaker in unseen_speakers:
    for sentence in unseen_sentences:
        source_file = os.path.join(source_directory, f'{speaker}/{speaker}_{sentence}_mic1.wav')
        final_destination_directory = f'{destination_directory}/{speaker}'
        os.makedirs(final_destination_directory, exist_ok=True)
        destination_file = os.path.join(destination_directory, f'{final_destination_directory}/{speaker}_{sentence}_mic1.wav')

        # 파일이 존재하는지 확인 후 복사
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            print(f"파일 복사 완료: {source_file} -> {destination_file}")
        else:
            print(f"경고: 파일이 존재하지 않습니다 - {source_file}")

print("복사 완료")