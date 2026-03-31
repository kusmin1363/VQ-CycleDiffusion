# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import random
import numpy as np
import torch
import tgt

from params import seed as random_seed
from params import n_mels, train_frames


def get_test_speakers():
    test_speakers = ['p236', 'p239', 'p259', 'p263']
    return test_speakers


def get_vctk_unseen_speakers():
    unseen_speakers =['p236', 'p239', 'p259', 'p263']
    return unseen_speakers


def get_vctk_unseen_sentences():
    unseen_sentences = ['002', '003', '004', '005', '006', '007', '009', '010', '011', '012',
                        '013', '014', '015', '016', '017', '018', '019', '020', '021', '023',
                        '024', '025', '026', '027', '028', '029', '030', '031', '032', '033'
                        ]
    unseen_sentences = []
    return unseen_sentences

def get_vctk_valid_sentences():
    unseen_sentences = ['002', '003', '004', '005', '006', '007', '009', '010', '011', '012',
                        '013', '014', '015', '016', '017', '018', '019', '020', '021', '023',
                        '024', '025', '026', '027', '028', '029', '030', '031', '032', '033'
                        ]
    unseen_sentences = []

    return unseen_sentences


# exclude utterances where MFA couldn't recognize some words
def exclude_spn(data_dir, spk, mel_ids):
    res = []
    for mel_id in mel_ids:
        textgrid = mel_id + '.TextGrid'
        tg_path = os.path.join(data_dir, 'textgrids', spk, textgrid)
        if not os.path.exists(tg_path):
            print(f"⚠️ TextGrid not found: {mel_id}")
            continue

        t = tgt.io.read_textgrid(os.path.join(data_dir, 'textgrids', spk, textgrid))
        t = t.get_tier_by_name('phones')
        spn_found = False
        for i in range(len(t)):
            if t[i].text == 'spn':
                spn_found = True
                break
        if not spn_found:
            res.append(mel_id)
    return res


class VCEncDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, avg_type):

        self.mel_x_dir = os.path.join(data_dir, 'mels')
        self.mel_y_dir = os.path.join(data_dir, 'mels_%s' % avg_type)

        self.test_speakers = get_test_speakers()
        self.speakers = os.listdir(self.mel_x_dir) #수정

        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            #mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        
        for spk in self.test_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        try : 
            mel_x_path = os.path.join(self.mel_x_dir, spk, mel_id + '_mel.npy')
            mel_y_path = os.path.join(self.mel_y_dir, spk, mel_id + '_avgmel.npy')
            mel_x = np.load(mel_x_path)
            mel_y = np.load(mel_y_path)
            mel_x = torch.from_numpy(mel_x).float()
            mel_y = torch.from_numpy(mel_y).float()
            return (mel_x, mel_y)
    
        except FileNotFoundError as e:
            print(f"[WARNING] Skipping missing file: {e}")
            return None

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y = self.get_vc_data(mel_id, spk)
        item = {'x': mel_x, 'y': mel_y}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        for mel_id, spk in self.test_info:
            result = self.get_vc_data(mel_id, spk)
            if result is not None:
                mel_x, mel_y = result  # ✅ 여기서만 unpack
                pairs.append((mel_x, mel_y))
        return pairs
    
# VCTK dataset for training "average voice" encoder
class VCTKEncDataset(torch.utils.data.Dataset):
    '''
    - The label of each mels file is the average mels 
    '''
    def __init__(self, data_dir, exc_file, avg_type):
        self.mel_x_dir = os.path.join(data_dir, 'mels')
        self.mel_y_dir = os.path.join(data_dir, 'mels_%s' % avg_type)

        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        # self.speakers = [spk for spk in os.listdir(self.mel_x_dir) 
        #                  if spk not in self.unseen_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        #train data
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            #mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        #test data
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] in self.unseen_sentences]
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.mel_x_dir, spk, mel_id + '_mel.npy')
        mel_y_path = os.path.join(self.mel_y_dir, spk, mel_id + '_avgmel.npy')
        mel_x = np.load(mel_x_path)
        mel_y = np.load(mel_y_path)
        mel_x = torch.from_numpy(mel_x).float()
        mel_y = torch.from_numpy(mel_y).float()
        return (mel_x, mel_y)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y = self.get_vc_data(mel_id, spk)
        item = {'x': mel_x, 'y': mel_y}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        for i in range(len(self.test_info)):
            mel_id, spk = self.test_info[i]
            mel_x, mel_y = self.get_vc_data(mel_id, spk)
            pairs.append((mel_x, mel_y))
        return pairs


class VCEncBatchCollate(object):
    '''
    - Get batch for training 
    - If the length of the mels_x is longer than the train_frames param
      we can choose a random start point and the mel_lenghts is the frame lenght
    - If the lenght of the mels_x is shorter than the train_frames param 
      the start point is 0 and the mel_lenghts is the lenght of x_mels
    '''
    def __call__(self, batch):
        B = len(batch)
        mels_x = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels_y = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['x'].shape[-1] - train_frames, 0) 
                      for item in batch]

        # if the length of the mels_x is longer than the train_frames param
        # we can choose a random start point  
        starts = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel_x = item['x']
            mel_y = item['y']
            if mel_x.shape[-1] < train_frames:
                mel_length = mel_x.shape[-1]
            else:
                mel_length = train_frames
            mels_x[i, :, :mel_length] = mel_x[:, starts[i]:starts[i] + mel_length]
            mels_y[i, :, :mel_length] = mel_y[:, starts[i]:starts[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        return {'x': mels_x, 'y': mels_y, 'lengths': mel_lengths}


# LibriTTS dataset for training speaker-conditional diffusion-based decoder
class VCDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val_file, exc_file):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.test_speakers = get_test_speakers()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.test_speakers]
        self.speakers = [spk for spk in self.speakers
                         if len(os.listdir(os.path.join(self.mel_dir, spk))) >= 10]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        with open(val_file) as f:
            valid_ids = f.readlines()
        self.valid_ids = set([v.strip() + '_mel.npy' for v in valid_ids])
        self.exceptions += self.valid_ids

        self.valid_info = [(v[:-8], v.split('_')[0]) for v in self.valid_ids]
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m not in self.exceptions]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        mels, embed = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((mels, embed))
        return pairs


# VCTK dataset for training speaker-conditional diffusion-based decoder
class VCTKDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file, mode):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk in self.unseen_speakers]
        self.mode = mode
        print(self.speakers)
        random.seed(random_seed)
        random.shuffle(self.speakers)
        
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]

        self.train_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m for m in mel_ids if m not in self.exceptions]            
            self.train_info += [(i[:-8], spk) for i in mel_ids]

        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    # def __getitem__(self, index):
    #     mels, embed = self.get_vc_data(self.train_info[index])
    #     item = {'mel': mels, 'c': embed}
    #     return item
    
    def __getitem__(self, index):
        data_folder = f'VCTK_2F2M_{self.mode}'

        src_mels, src_embed = self.get_vc_data(self.train_info[index])

        src_audio, src_spk = self.train_info[index]

        tgt_list = self.unseen_speakers.copy()
        tgt_list.remove(src_spk)

        tgt_spk = random.choice(tgt_list)
        
        folder_path = f'{data_folder}/mels/{tgt_spk}'
        
        tgt_sentences = []
        tgt_mels_list = []
        tgt_embed_list = []

        for item in os.listdir(folder_path):
            sentence = item.split("_")
            tgt_sentences.append(sentence[1])
        
        tgt_sentence = random.choice(tgt_sentences)
        tgt_audio = src_audio.split("_")
        tgt_audio[0] = tgt_spk
        tgt_audio[1] = tgt_sentence
        tgt_audio = "_".join(tgt_audio)
        tgt_mels, tgt_embed = self.get_vc_data((tgt_audio, tgt_spk))
        #print("#####")
        #print("src:", src_audio)
        #print("tgt:", tgt_audio)
        item = {'mel': src_mels, 'c': src_embed, 'tgt_mel': tgt_mels, 'tgt_c': tgt_embed}
        return item
    
    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append({'mel': mels, 'c': embed})
        return pairs


class VCDecBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
         
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)


        mels_target = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts2 = [max(item['tgt_mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts2]
        mel_target_lengths = []
        for i, item in enumerate(batch):
            mel = item['tgt_mel']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels_target[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_target_lengths.append(mel_length)
        mel_target_lengths = torch.LongTensor(mel_target_lengths)
        tgt_embed = torch.stack([item['tgt_c'] for item in batch], 0)



        return {'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths, 'c': embed, 'mel_tgt': mels_target, 'tgt_mel_lengths': mel_target_lengths, 'tgt_c': tgt_embed}


# VCTK dataset for training speaker-conditional diffusion-based decoder
class VCTKDecDatasetMapping(torch.utils.data.Dataset):
    def __init__(self, data_dir, exc_file):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]

        self.train_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m for m in mel_ids if m not in self.exceptions]            
            self.train_info += [(i[:-8], spk) for i in mel_ids]

        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed
    
    def __getitem__(self, index):
        data_folder = 'VCTK_2F2M'
        src_mels, src_embed = self.get_vc_data(self.train_info[index])

        src_audio, src_spk = self.train_info[index]

        tgt_list = self.unseen_speakers.copy()
        tgt_list.remove(src_spk)

        tgt_spk = random.choice(tgt_list)
        
        folder_path = f'{data_folder}/mels/{tgt_spk}'
        
        tgt_sentences = []
        tgt_mels_list = []
        tgt_embed_list = []

        for item in os.listdir(folder_path):
            sentence = item.split("_")
            tgt_sentences.append(sentence[1])
        
        tgt_sentence = random.choice(tgt_sentences)
        tgt_audio = src_audio.split("_")
        tgt_audio[0] = tgt_spk
        tgt_audio[1] = tgt_sentence
        tgt_audio = "_".join(tgt_audio)
        tgt_mels, tgt_embed = self.get_vc_data((tgt_audio, tgt_spk))
        #print("#####")
        #print("src:", src_audio)
        #print("tgt:", tgt_audio)

        item = {'mel': src_mels, 'c': src_embed, 'tgt_mel': tgt_mels, 'tgt_c': tgt_embed,  
                'src_spk': src_spk, 'tgt_spk': tgt_spk}
        return item
    
    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            # getitem 로직과 유사하게 소스/타겟 쌍을 생성합니다.
            src_mels, src_embed = self.get_vc_data(self.valid_info[i])
            src_audio, src_spk = self.valid_info[i]

            # 타겟 화자 랜덤 선택
            tgt_list = self.unseen_speakers.copy()
            tgt_list.remove(src_spk)
            tgt_spk = random.choice(tgt_list)
            
            # 타겟 화자의 발화 중 하나를 랜덤으로 선택 (getitem 로직과 동일)
            folder_path = os.path.join(self.mel_dir, tgt_spk)
            tgt_utterances = [f[:-8] for f in os.listdir(folder_path)]
            tgt_audio = random.choice(tgt_utterances)
            
            tgt_mels, tgt_embed = self.get_vc_data((tgt_audio, tgt_spk))
            
            # 학습 데이터와 동일한 형식으로 저장
            pairs.append({'mel': src_mels, 'c': src_embed,
                          'tgt_mel': tgt_mels, 'tgt_c': tgt_embed,
                          'src_spk': src_spk, 'tgt_spk': tgt_spk})
        return pairs
    


class VCDecBatchCollateMapping(object):
    def __call__(self, batch):
        B = len(batch)
         
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)


        mels_target = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts2 = [max(item['tgt_mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts2]
        mel_target_lengths = []
        for i, item in enumerate(batch):
            mel = item['tgt_mel']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels_target[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_target_lengths.append(mel_length)
        mel_target_lengths = torch.LongTensor(mel_target_lengths)
        tgt_embed = torch.stack([item['tgt_c'] for item in batch], 0)

        # 각 item에서 'src_spk'와 'tgt_spk'를 추출하여 리스트로 만듭니다.
        src_spks = [item['src_spk'] for item in batch]
        tgt_spks = [item['tgt_spk'] for item in batch]
        # --------------------

        # --- [수정된 부분] ---
        # 최종 반환 딕셔너리에 화자 ID 리스트를 추가합니다.
        return {'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths, 'c': embed, 
                'mel_tgt': mels_target, 'tgt_mel_lengths': mel_target_lengths, 'tgt_c': tgt_embed,
                'src_spk': src_spks, 'tgt_spk': tgt_spks}