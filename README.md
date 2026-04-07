# 실험 진행 방법
📂 VQ-CycleDiffusion Project Structure
1. Baseline & Data Setup

Baseline Scripts: train_cyclediffusion_enc.py, train_cyclediffusion_dec.py

Dataset Split:

Test Data (30 speakers): ['002' ~ '033'] (008, 022 제외)

Valid Data (30 speakers): ['034' ~ '063']

Pre-trained Model:

2. Training Scripts (train/ folder)
학습 목적과 범위에 따라 크게 5가지 코드로 분류됩니다. (indv: 특정 화자, global: 전체 화자)

init_codebook_stock (indv/global): CycleDiffusion MelEncoder의 출력값(Encoded Output)을 기반으로 K-Means 클러스터링을 통해 초기 Codebook 생성

train_codebook_only (indv/global): Encoder - Codebook - Decoder 연결 후 Codebook 파라미터만 단독 학습

train_decoder (cycle/recon): Encoder - Codebook - Decoder 연결 후 Cycle loss 또는 Recon loss를 사용하여 Decoder만 학습

train_codebook_decoder_joint: Encoder - Codebook - Decoder 연결 후 Codebook과 Decoder를 동시에 학습

train_codebook_all_joint (indv/global): Encoder, Codebook, Decoder 전체 네트워크를 End-to-End로 학습

3. Utility & Inference Scripts

Mapping: counting_map_script (Count Map을 활용하는 변환 방식의 경우, Speaker2Speaker 또는 Speaker2Global 형태의 Count Map 생성)

Conversion: convert/make_convert 또는 convert/make_convert_vq_train 사용

내부 지원 로직: Original CycleDiffusion 변환, Global Codebook 단독 활용 변환, VQ-CycleDiffusion 방법론

Inference: convert/inference, convert/inference_vq_train

Evaluation (MCD): calculate/ 폴더 내 cal_pymcd, cal_pymcd_single 등 사용
