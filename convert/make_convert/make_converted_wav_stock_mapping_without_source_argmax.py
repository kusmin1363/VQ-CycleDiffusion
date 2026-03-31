import os
from convert.inference.inference_vq_mapping_without_source_argmax import inference
from itertools import permutations
from tqdm import tqdm
import torch


# unseen_sentences = ['002', '003', '004', '005', '006', '007', '009', '010', '011', '012',
#                     '013', '014', '015', '016', '017', '018', '019', '020', '021', '023',
#                     '024', '025', '026', '027', '028', '029', '030', '031', '032', '033'
#                     ]

unseen_sentences = ['034', '035', '036', '037', '038', '039', '040', '041', '042', '043',
                    '044', '045', '046', '047', '048', '049', '050', '051', '052', '053',
                    '054', '055', '056', '057', '058', '059', '060', '061', '062', '063'
                    ]

seen_speakers = ['p236','p239',  'p259','p263']

import argparse

parser = argparse.ArgumentParser(description='T')
parser.add_argument('--size', type=int, default=2048)

argv = parser.parse_args()
seenORunseen = 'seen'
speakers = seen_speakers if seenORunseen == 'seen' else unseen_speakers
test_path = 'VCTK_2F2M_valid/wavs'

codebooksize = argv.size

model_name = f'vq_{codebooksize}' 
output_dir = f'final_samples_fixed/vq_mapping/global_tgt_argmax/{model_name}'
vc_model_path = f'log/log_Gunhee/vc_255.pt'
output_dir = os.path.join(output_dir)

for src_speaker, tgt_speaker in permutations(speakers, 2):
    for src_sentence, tgt_sentence in permutations(unseen_sentences, 2):
        src_path = os.path.join(test_path, f'{src_speaker}', f'{src_speaker}_{src_sentence}.wav')
        tgt_path = os.path.join(test_path,f'{tgt_speaker}', f'{tgt_speaker}_{tgt_sentence}.wav')
        
        output_path = os.path.join(output_dir, f'{src_speaker}_to_{tgt_speaker}', f'{src_sentence}_to_{tgt_sentence}.wav')
        if not os.path.exists(output_path):
            # 폴더가 없을 경우 생성
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))        

            # 파일이 존재하는지 확인 후 inference 호출
            if os.path.exists(src_path) and os.path.exists(tgt_path):
                #torch.cuda.empty_cache()
                inference(vc_model_path, src_path, tgt_path, output_path, codebooksize)
                print(f"{src_speaker}_to_{tgt_speaker} and {src_sentence}_to_{tgt_sentence} complete")
            else:
                continue
        else:
            continue

