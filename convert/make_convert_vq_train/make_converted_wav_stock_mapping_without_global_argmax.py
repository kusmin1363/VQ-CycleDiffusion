#python3 -m convert.make_convert_vq_train.make_converted_wav_stock_mapping_without_global_argmax1 --size 512 --decoder_base_path "log/Decoder_cycle_only_10.0/local/512_diff_1e-08"

import os
import argparse
from itertools import permutations
from convert.inference_vq_train.inference_vq_mapping_without_global_argmax import inference

unseen_sentences = [f'{i:03d}' for i in range(34, 64)] 
unseen_sentences = ['002', '003', '004', '005', '006', '007', '009', '010', '011', '012',
                    '013', '014', '015', '016', '017', '018', '019', '020', '021', '023',
                    '024', '025', '026', '027', '028', '029', '030', '031', '032', '033'
                    ]

seen_speakers = ['p236', 'p239', 'p259', 'p263']

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_path', type=str, default="")
parser.add_argument('--codebook_base_path', type=str, default="")
parser.add_argument('--decoder_base_path', type=str, default="")
parser.add_argument('--map_base_path', type=str, default="")
parser.add_argument('--size', type=int, default=255)


args = parser.parse_args()

enc_path = args.encoder_path if args.encoder_path else None
cb_path = args.codebook_base_path if args.codebook_base_path else None
dec_path = args.decoder_base_path if args.decoder_base_path else None
map_path = args.map_base_path if args.map_base_path else None


enc_flag = "True" if enc_path else "False"
cb_flag = "True" if cb_path else "False"
dec_flag = "True" if dec_path else "False"

test_path = 'VCTK_2F2M_valid/wavs'
speakers = seen_speakers  

model_name = f'enc_{enc_flag}_cb_{cb_flag}_dec_{dec_flag}_v2'
output_dir = f'final_samples/test/src_tgt_argmax/{model_name}'

for src_speaker, tgt_speaker in permutations(speakers, 2):
    for src_sentence, tgt_sentence in permutations(unseen_sentences, 2):
        src_path = os.path.join(test_path, src_speaker, f'{src_speaker}_{src_sentence}.wav')
        tgt_path = os.path.join(test_path, tgt_speaker, f'{tgt_speaker}_{tgt_sentence}.wav')
        
        output_path = os.path.join(output_dir, f'{src_speaker}_to_{tgt_speaker}', f'{src_sentence}_to_{tgt_sentence}.wav')
        
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))        

                    
        if os.path.exists(src_path) and os.path.exists(tgt_path) and not os.path.exists(output_path):
            inference(
                        src_speaker=src_speaker, 
                        tgt_speaker=tgt_speaker,
                        src_wav_path=src_path, 
                        tgt_wav_path=tgt_path, 
                        output_path=output_path, 
                        codebooksize=args.size,

                        encoder_path=enc_path,
                        codebook_base_path=cb_path,
                        decoder_base_path=dec_path,
                        map_base_path=map_path
                    )

            print(f"[{model_name}] {src_speaker}_{src_sentence} -> {tgt_speaker}_{tgt_sentence} Complete")