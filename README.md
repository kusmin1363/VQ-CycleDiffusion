# 실험 진행 방법
Baseline 생성
1. train_cyclediffusion_enc.py
2. train_cyclediffusion_dec.py

test_data : ['002', '003', '004', '005', '006', '007', '009', '010', '011', '012',
            '013', '014', '015', '016', '017', '018', '019', '020', '021', '023',
            '024', '025', '026', '027', '028', '029', '030', '031', '032', '033']

valid_data : ['034', '035', '036', '037', '038', '039', '040', 
            '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', 
            '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', 
            '061', '062', '063' ]
            
pre-trained 모델(Decoder) :
Dataset : 

이를 바탕으로 train 폴더 내에 있는 코드 진행
직접 사용하는 코드는 크게 5가지 종류가 있음 (indv : 특정 화자의 데이터만, global : 전체 화자의 데이터)
  1. init_codebook_stock (indv/global) - CycleDiffusion의 MelEncoder의 출력값(Encoded Output)을 바탕으로 Codebook 생성(K-Means)
  2. train_codebook_only (global / indv) - Encoder - Codebook - Decoder 구조 연결한 이후, Codebook 만 학습
  3. train_decoder (cycle / recon) - Encoder - Codebook - Decoder 구조 연결한 이후, cycle loss 혹은 recon loss를 이용해서 Decoder만 학습 진행
  4. train_codebook_decoder_joint - Encoder - Codebook - Decoder 구조 연결한 이후, Codebook, Decoder를 동시에 학습
  5. train_codebook_all_joint (global/indv) - encoder, codebook, decoder를 모두 학습

그 외에 사용하는 코드
  - counting_map_script : Count Map을 사용하는 변환 방식의 경우, Speaker2Speaker, Speaker2Global 형태로 Count Map 생성 가능

변환할 때는 convert에 들어있는 코드 사용. 정확하게는 convert/make_convert 혹은 conver/make_convert_vq_train 2가지 중에서 골라서 사용
  - 내부에는 Original CycleDiffusion 변환 코드, Global Codebook만을 사용한 변환 코드, VQ-CycleDiffusion의 방법론 코드가 존재
  - inference 코드는 convert/inference, convert/inference_vq_train 에서 확인 가능

MCD 측정은 calculate 폴더에서, cal_pymcd, cal_pymcd_single 등 다양하게 사용
