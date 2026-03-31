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
from itertools import permutations

def get_test_speakers():
    test_speakers = ['1401', '2238', '3723', '4014', '5126', 
                     '5322', '587', '6415', '8057', '8534']
    return test_speakers


def get_vctk_unseen_speakers():
    # 4명으로 many-to-many 이므로 unseen speaker는 없다.
    unseen_speakers = []
    return unseen_speakers


def get_vctk_unseen_sentences():
    # 화자 당 10개의 test sentences
    unseen_sentences = ['002', '003', '004', '005', '006', '007', '009', '010', '011', '012']
    return unseen_sentences


# exclude utterances where MFA couldn't recognize some words
def exclude_spn(data_dir, spk, mel_ids):
    res = []
    for mel_id in mel_ids:
        textgrid = mel_id + '.TextGrid'
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



# VCTK dataset for training speaker-conditional diffusion-based decoder
class VCTKDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
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

        tgt_list = self.speakers.copy()
        tgt_list.remove(src_spk)

        tgt_spk = random.choice(tgt_list)
        
        folder_path = f'/home/smin1363/speechst2/real/CycleDiffusion/{data_folder}/mels/{tgt_spk}'
        
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
            pairs.append((mels, embed))
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

    