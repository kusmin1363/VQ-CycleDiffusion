from os import stat
import numpy as np
import torch
import math
import gc
import itertools
from speech_tools import world_decode_mc, world_speech_synthesis
import soundfile
import os
import time
import subprocess
from preprocess import preprocess_tools

def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    
    # Logarithm Gaussian normalization for Pitch Conversions
    f0_normalized_t = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_normalized_t

########################

def split_train_dev(datadict, train_percent=0.8):
    train_dict = dict()
    dev_dict = dict()
    for spk_id, cur_data in datadict.items():
        datanum = len(cur_data)
        train_num = int(datanum*train_percent)
        train_dict[spk_id]=cur_data[:train_num]
        dev_dict[spk_id]=cur_data[train_num:]
    return train_dict, dev_dict

def extract_target_from_ppg(ppg_mat, window=4):
    n_frames = ppg_mat.shape[-1]
    assert n_frames % window == 0, str(n_frames)+"\t"+str(window)
    target_list = []
    for start_idx in range(0, n_frames, window):
        end_idx = start_idx + window
        
        cur_ppg = np.sum(ppg_mat[:, start_idx:end_idx], axis=1)
        target_idx = np.argmax(cur_ppg)
        target_list.append(target_idx)
    
    return target_list

def sample_train_data(sp_list, n_frames=128, shuffle=False, ppg_list=None):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """

    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    target_list = []

    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_data_frames = cur_data.shape[-1]
        
        assert cur_data_frames >= n_frames, "Too short SP"
        
        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        # start_idx = np.random.randint(cur_data_frames - n_frames)
        end_idx = start_idx + n_frames

        cur_sp_mat = cur_data[:, start_idx:end_idx]
        sp_mat_list.append(cur_sp_mat)
        if ppg_list is not None:
            cur_ppg = ppg_list[idx]
            cur_ppg_mat = cur_ppg[:, start_idx:end_idx]
            cur_target = extract_target_from_ppg(cur_ppg_mat, window=4)
            target_list.append(cur_target)

    result = np.array(sp_mat_list)
    targets = None if ppg_list == None else np.array(target_list)
    return result, targets

def sample_train_data2(sp_list, n_frames=128, shuffle=False, ppg_list=None, phone_list=None, voted=False):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """

    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    target_list = []
    target_list_phone = []

    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_data_frames = cur_data.shape[-1]
        cur_phone = phone_list[idx]

        assert cur_data_frames >= n_frames, "Too short SP"
        
        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        # start_idx = np.random.randint(cur_data_frames - n_frames)
        end_idx = start_idx + n_frames

        cur_sp_mat = cur_data[:, start_idx:end_idx]
        sp_mat_list.append(cur_sp_mat)

        if ppg_list is not None:
            cur_ppg = ppg_list[idx]
            cur_ppg_mat = cur_ppg[:, start_idx:end_idx]
            cur_target = extract_target_from_ppg(cur_ppg_mat, window=4)
            target_list.append(cur_target)

        if phone_list is not None:
            phone_len = len(cur_phone)
            #print("Total Num = {}, Phone Len = {}".format(cur_data_frames, phone_len))
            if cur_data_frames > phone_len :
                phone_map_list = []
                ratio = round(phone_len / cur_data_frames, 2)
                phone_start_idx = math.trunc(round(start_idx / cur_data_frames, 2) * phone_len)
                phone_frame = math.trunc(ratio * n_frames)
                phone_end_idx = phone_start_idx + phone_frame
                if phone_end_idx > phone_len : 
                    phone_end_idx = phone_len

                left_frame = n_frames - (phone_end_idx - phone_start_idx)

                if left_frame == 0 :
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    type = 0

                if left_frame < 0 : 
                    phone_map_list.extend(cur_phone[phone_end_idx-128:phone_end_idx])
                    type = 1

                if left_frame > 0 and left_frame % 2 == 0 : 
                    type = 2
                    for i in range(math.trunc(left_frame/2)) :
                        phone_map_list.append(cur_phone[phone_start_idx])
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    for i in range(math.trunc(left_frame/2)) :
                        assert (phone_end_idx) <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                        phone_map_list.append(cur_phone[phone_end_idx-1])

                elif left_frame > 0 and left_frame % 2 != 0:
                    type = 3
                    for i in range(math.trunc(left_frame/2)) :
                        phone_map_list.append(cur_phone[phone_start_idx])
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    for i in range(math.trunc(left_frame/2) + 1) :
                        assert (phone_end_idx) <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                        phone_map_list.append(cur_phone[phone_end_idx-1])
                
                assert len(phone_map_list) == 128, "Wrong Length0! : {}, type : {}".format(len(phone_map_list), type)
                if voted==True : 
                    voted_phone_list = []
                    for i in range(0, len(phone_map_list), 4) : 
                        vote = np.zeros(196)
                        window = phone_map_list[i:i+4]
                        vote[int(window[0])] += 1
                        vote[int(window[1])] += 1
                        vote[int(window[2])] += 1
                        vote[int(window[3])] += 1
                        max_idx = np.argmax(vote)
                        max_val = np.max(vote)
                        voted_phone_list.append(max_idx)
                        del vote
                        gc.collect()
                    assert len(voted_phone_list) == 32, "\nWrong Voted Phone length! : {}".format(len(voted_phone_list))

                    target_list_phone.append(voted_phone_list)
                else :
                    target_list_phone.append(phone_map_list)

            else : 
                phone_map_list = []
                phone_start_idx = math.trunc(round(start_idx / cur_data_frames, 2) * phone_len)
                phone_end_idx = phone_start_idx + n_frames
                if phone_end_idx > phone_len : 
                    phone_end_idx = phone_len
                left_frame = n_frames - (phone_end_idx - phone_start_idx)

                if left_frame == 0 :
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    type = 0

                if left_frame < 0 : 
                    phone_map_list.extend(cur_phone[phone_end_idx-128:phone_end_idx])
                    type = 1

                if left_frame > 0 and left_frame % 2 == 0 : 
                    type = 2
                    for i in range(math.trunc(left_frame/2)) :
                        phone_map_list.append(cur_phone[phone_start_idx])
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    for i in range(math.trunc(left_frame/2)) :
                        assert (phone_end_idx) <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                        phone_map_list.append(cur_phone[phone_end_idx-1])
                elif left_frame > 0 and left_frame % 2 != 0:
                    type = 3
                    for i in range(math.trunc(left_frame/2)) :
                        phone_map_list.append(cur_phone[phone_start_idx])
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    for i in range(math.trunc(left_frame/2) + 1) :
                        assert phone_end_idx <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                        phone_map_list.append(cur_phone[phone_end_idx-1])

                assert len(phone_map_list) == 128, "Wrong Length0! : {}, type : {}".format(len(phone_map_list), type)
                if voted==True : 
                    voted_phone_list = []
                    for i in range(0, len(phone_map_list), 4) : 
                        vote = np.zeros(196)
                        window = phone_map_list[i:i+4]
                        vote[int(window[0])] += 1
                        vote[int(window[1])] += 1
                        vote[int(window[2])] += 1
                        vote[int(window[3])] += 1
                        max_idx = np.argmax(vote)
                        max_val = np.max(vote)
                        voted_phone_list.append(max_idx)
                        del vote
                        gc.collect()
                    assert len(voted_phone_list) == 32, "\nWrong Voted Phone length! : {}".format(len(voted_phone_list))

                    target_list_phone.append(voted_phone_list)
                else :
                    target_list_phone.append(phone_map_list)


    result = np.array(sp_mat_list)
    targets = None if ppg_list == None else np.array(target_list)
    src_phone = None if phone_list == None else np.array(target_list_phone)
    return result, targets, src_phone

def sample_train_data3(sp_list, n_frames=128, shuffle=False, ppg_list=None, phone_list=None, voted=False):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """

    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    target_list = []
    target_list_phone = []

    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_data_frames = cur_data.shape[-1]
        cur_phone = phone_list[idx]

        assert cur_data_frames >= n_frames, "Too short SP"
        
        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        # start_idx = np.random.randint(cur_data_frames - n_frames)
        end_idx = start_idx + n_frames

        cur_sp_mat = cur_data[:, start_idx:end_idx]
        sp_mat_list.append(cur_sp_mat)

        if ppg_list is not None:
            cur_ppg = ppg_list[idx]
            cur_ppg_mat = cur_ppg[:, start_idx:end_idx]
            cur_target = extract_target_from_ppg(cur_ppg_mat, window=4)
            target_list.append(cur_target)

        if phone_list is not None:
            phone_len = len(cur_phone)
            #print("Total Num = {}, Phone Len = {}".format(cur_data_frames, phone_len))
            if cur_data_frames > phone_len :
                phone_map_list = []
                ratio = round(phone_len / cur_data_frames, 2)
                phone_start_idx = math.trunc(round(start_idx / cur_data_frames, 2) * phone_len)
                phone_frame = math.trunc(ratio * n_frames)
                phone_end_idx = phone_start_idx + phone_frame
                if phone_end_idx > phone_len : 
                    phone_end_idx = phone_len

                left_frame = n_frames - (phone_end_idx - phone_start_idx)

                if left_frame == 0 :
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    type = 0

                if left_frame < 0 : 
                    phone_map_list.extend(cur_phone[phone_end_idx-128:phone_end_idx])
                    type = 1

                if left_frame > 0 and left_frame % 2 == 0 : 
                    type = 2
                    iter_num = math.trunc(left_frame / phone_frame)
                    if iter_num <= 0 : 
                        for i in range(math.trunc(left_frame/2)) :
                            phone_map_list.append(cur_phone[phone_start_idx])
                        phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                        for i in range(math.trunc(left_frame/2)) :
                            assert (phone_end_idx) <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                            phone_map_list.append(cur_phone[phone_end_idx-1])
                    else : 
                        tmp = list(itertools.chain.from_iterable(itertools.repeat(x, iter_num) for x in cur_phone[phone_start_idx:phone_end_idx]))
                        left_frame2 = n_frames - len(tmp)
                        
                        if left_frame2 == 0 :
                            phone_map_list.extend(tmp)
                        if left_frame2 < 0 : 
                            phone_map_list.extend(tmp[len(tmp)-n_frames+1 : -1])
                        if left_frame2 > 0 and left_frame2 % 2 == 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[-1])
                        elif left_frame2 > 0 and left_frame2 % 2 != 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2) + 1) :
                                phone_map_list.append(tmp[-1])


                elif left_frame > 0 and left_frame % 2 != 0:
                    type = 3
                    iter_num = math.trunc(left_frame / phone_frame)
                    if iter_num <= 0 : 
                        for i in range(math.trunc(left_frame/2)) :
                            phone_map_list.append(cur_phone[phone_start_idx])
                        phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                        for i in range(math.trunc(left_frame/2) + 1) :
                            assert (phone_end_idx) <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                            phone_map_list.append(cur_phone[phone_end_idx-1])
                    else : 
                        tmp = list(itertools.chain.from_iterable(itertools.repeat(x, iter_num) for x in cur_phone[phone_start_idx:phone_end_idx]))
                        left_frame2 = n_frames - len(tmp)
                        
                        if left_frame2 == 0 :
                            phone_map_list.extend(tmp)
                        if left_frame2 < 0 : 
                            phone_map_list.extend(tmp[len(tmp)-n_frames+1 : -1])
                        if left_frame2 > 0 and left_frame2 % 2 == 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[-1])
                        elif left_frame2 > 0 and left_frame2 % 2 != 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2) + 1) :
                                phone_map_list.append(tmp[-1])
                
                assert len(phone_map_list) == 128, "Wrong Length0! : {}, type : {}".format(len(phone_map_list), type)
                
                if voted==True : 
                    voted_phone_list = []
                    for i in range(0, len(phone_map_list), 4) : 
                        vote = np.zeros(196)
                        window = phone_map_list[i:i+4]
                        vote[int(window[0])] += 1
                        vote[int(window[1])] += 1
                        vote[int(window[2])] += 1
                        vote[int(window[3])] += 1
                        max_idx = np.argmax(vote)
                        max_val = np.max(vote)
                        voted_phone_list.append(max_idx)
                        del vote
                        gc.collect()
                    assert len(voted_phone_list) == 32, "\nWrong Voted Phone length! : {}".format(len(voted_phone_list))

                    target_list_phone.append(voted_phone_list)
                else :
                    target_list_phone.append(phone_map_list)

            else : 
                phone_map_list = []
                phone_start_idx = math.trunc(round(start_idx / cur_data_frames, 10) * phone_len)
                phone_end_idx = phone_start_idx + n_frames
                if phone_end_idx > phone_len : 
                    phone_end_idx = phone_len
                left_frame = n_frames - (phone_end_idx - phone_start_idx)

                if left_frame == 0 :
                    phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                    type = 0

                if left_frame < 0 : 
                    phone_map_list.extend(cur_phone[phone_end_idx-128:phone_end_idx])
                    type = 1

                if left_frame > 0 and left_frame % 2 == 0 : 
                    type = 2
                    iter_num = math.trunc(left_frame / phone_frame)
                    if iter_num <= 0 : 
                        for i in range(math.trunc(left_frame/2)) :
                            phone_map_list.append(cur_phone[phone_start_idx])
                        phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                        for i in range(math.trunc(left_frame/2)) :
                            assert (phone_end_idx) <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                            phone_map_list.append(cur_phone[phone_end_idx-1])
                    else : 
                        tmp = list(itertools.chain.from_iterable(itertools.repeat(x, iter_num) for x in cur_phone[phone_start_idx:phone_end_idx]))
                        left_frame2 = n_frames - len(tmp)
                        
                        if left_frame2 == 0 :
                            phone_map_list.extend(tmp)
                        if left_frame2 < 0 : 
                            phone_map_list.extend(tmp[len(tmp)-n_frames+1 : -1])
                        if left_frame2 > 0 and left_frame2 % 2 == 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[-1])
                        elif left_frame2 > 0 and left_frame2 % 2 != 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2) + 1) :
                                phone_map_list.append(tmp[-1])


                elif left_frame > 0 and left_frame % 2 != 0:
                    type = 3
                    iter_num = math.trunc(left_frame / phone_frame)
                    if iter_num <= 0 : 
                        for i in range(math.trunc(left_frame/2)) :
                            phone_map_list.append(cur_phone[phone_start_idx])
                        phone_map_list.extend(cur_phone[phone_start_idx:phone_end_idx])
                        for i in range(math.trunc(left_frame/2) + 1) :
                            assert (phone_end_idx) <= len(cur_phone), "start_idx : {}, ratio : {}, \nphone_start_idx : {}, phone_end_idx : {}, frames : {}, len : {}".format(start_idx, ratio, phone_start_idx, phone_end_idx-1, phone_frame, len(cur_phone))
                            phone_map_list.append(cur_phone[phone_end_idx-1])
                    else : 
                        tmp = list(itertools.chain.from_iterable(itertools.repeat(x, iter_num) for x in cur_phone[phone_start_idx:phone_end_idx]))
                        left_frame2 = n_frames - len(tmp)
                        
                        if left_frame2 == 0 :
                            phone_map_list.extend(tmp)
                        if left_frame2 < 0 : 
                            phone_map_list.extend(tmp[len(tmp)-n_frames+1 : -1])
                        if left_frame2 > 0 and left_frame2 % 2 == 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[-1])
                        elif left_frame2 > 0 and left_frame2 % 2 != 0 : 
                            for i in range(math.trunc(left_frame2/2)) :
                                phone_map_list.append(tmp[0])
                            phone_map_list.extend(tmp)
                            for i in range(math.trunc(left_frame2/2) + 1) :
                                phone_map_list.append(tmp[-1])

                assert len(phone_map_list) == 128, "Wrong Length1! : {}, Type : {}".format(len(phone_map_list), type)

                if voted==True : 
                    voted_phone_list = []
                    for i in range(0, len(phone_map_list), 4) : 
                        vote = np.zeros(196)
                        window = phone_map_list[i:i+4]
                        vote[int(window[0])] += 1
                        vote[int(window[1])] += 1
                        vote[int(window[2])] += 1
                        vote[int(window[3])] += 1
                        max_idx = np.argmax(vote)
                        max_val = np.max(vote)
                        voted_phone_list.append(max_idx)
                        del vote
                        gc.collect()
                    assert len(voted_phone_list) == 32, "\nWrong Voted Phone length! : {}".format(len(voted_phone_list))

                    target_list_phone.append(voted_phone_list)
                else :
                    target_list_phone.append(phone_map_list)

    result = np.array(sp_mat_list)
    targets = None if ppg_list == None else np.array(target_list)
    src_phone = None if phone_list == None else np.array(target_list_phone)
    return result, targets, src_phone

def sample_train_data4(sp_list, n_frames=128, shuffle=False, ppg_list=None, f0s=None, sps=None, aps=None, voted=False, is_phone=True):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """
    ctm_path_4spk = "/home/klklp98/speechst2/zeroth_n/zeroth/s5/EXTRACT2/phones.ctm"
    flac_path_4spk = "/home/klklp98/speechst2/Exp_Disentanglement/for_LI2/phone_check.flac"
    ctm_path_10spk = "/home/klklp98/speechst2/zeroth_n/zeroth/s5/EXTRACT/phones.ctm"
    flac_path_10spk = "/home/klklp98/speechst2/Exp_Disentanglement/for_LI/phone_check.flac"
    frame_shift = 0.01
    SIL_set = set([1, 2, 3, 4, 5])
    SPN_set = set([0, 6, 7, 8, 9])

    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    f0_list = []
    n_sp_list = []
    ap_list = []
    if is_phone :
        phone_list = []

    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_f0 = f0s[idx]
        cur_sp = sps[idx]
        cur_ap = aps[idx]

        cur_data_frames = cur_data.shape[-1]

        assert cur_data_frames >= n_frames, "Too short SP"
        
        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        # start_idx = np.random.randint(cur_data_frames - n_frames)
        end_idx = start_idx + n_frames

        cur_sp_mat = cur_data[:, start_idx:end_idx]
        cur_f0 = cur_f0[start_idx:end_idx]
        cur_sp = cur_sp[start_idx:end_idx, :]
        cur_ap = cur_ap[start_idx:end_idx, :]
        
        sp_mat_list.append(cur_sp_mat)
        f0_list.append(cur_f0)
        n_sp_list.append(cur_sp)
        ap_list.append(cur_ap)

    if is_phone :
        # kaldi 이용해서 phone 추출 시작
        for i in range(len(f0_list)) : 
            print("TOTAL LIST NUM : {}/{}".format(i, len(f0_list)))
            wav_transformed_mean = world_speech_synthesis(f0=f0_list[i], decoded_sp=n_sp_list[i], 
                ap=ap_list[i], fs=16000, frame_period=5.0)
            wav_transformed_mean = np.nan_to_num(wav_transformed_mean)
            '''
            # 4spk
            soundfile.write(flac_path_4spk, wav_transformed_mean, 16000)
            time.sleep(1)
            cmd = os.path.join(os.getcwd(), "for_LI2/LI_decode.sh")
            subprocess.run([cmd, flac_path_4spk], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            #os.system('{} {}'.format(cmd, flac_path_4spk))
            time.sleep(1)
            with open(ctm_path_4spk, "r") as ctm : 
                cur_phone = []
                for l in ctm.readlines() :
                    num_of_frames = int(float(l.split(' ')[3]) / frame_shift)
                    phone = l.split(' ')[4]
                    #print(phone)

                    # prefix를 제거하고 phone_list 추가
                    if int(phone) in SIL_set : 
                        for i in range(num_of_frames) : 
                            cur_phone.append(0)
                    elif int(phone) in SPN_set : 
                        assert True == False , "There is SPN phone in {}".format(l)
                        #minus_list.append(l)
                    else : 
                        phone = int(phone)
                        phone -= 11
                        phone = (phone // 4) + 2
                        if phone < 0 :
                            assert True == False , "phone is minus"
                        for i in range(num_of_frames) : 
                            cur_phone.append(phone)
            
            ############################################################################
            '''
            # 10spk
            soundfile.write(flac_path_10spk, wav_transformed_mean, 16000)
            time.sleep(1)
            cmd = os.path.join(os.getcwd(), "for_LI/LI_decode.sh")
            subprocess.run([cmd, flac_path_10spk], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            #os.system('{} {}'.format(cmd, flac_path_10spk))
            time.sleep(1)
            with open(ctm_path_10spk, "r") as ctm : 
                cur_phone = []
                for l in ctm.readlines() :
                    num_of_frames = int(float(l.split(' ')[3]) / frame_shift)
                    phone = l.split(' ')[4]
                    #print(phone)

                    # prefix를 제거하고 phone_list 추가
                    if int(phone) in SIL_set : 
                        for i in range(num_of_frames) : 
                            cur_phone.append(0)
                    elif int(phone) in SPN_set : 
                        assert True == False , "There is SPN phone in {}".format(l)
                        #minus_list.append(l)
                    else : 
                        phone = int(phone)
                        phone -= 11
                        phone = (phone // 4) + 2
                        if phone < 0 :
                            assert True == False , "phone is minus"
                        for i in range(num_of_frames) : 
                            cur_phone.append(phone)
            

            #print("CUR PHONE : {}".format(cur_phone))
            if len(cur_phone) < 32 : 
                left = 32 - len(cur_phone)
                pre_left = int(left / 2)
                post_left = left - pre_left
                first = cur_phone[0]
                last = cur_phone[-1]
                for i in range(pre_left) : 
                    cur_phone.insert(0, first)
                for i in range(post_left) : 
                    cur_phone.append(last)
            elif len(cur_phone) > 32 : 
                cur_phone = cur_phone[:32]

            assert len(cur_phone) == 32, "phone len is wrong ... {}".format(len(cur_phone))
            phone_list.append(cur_phone)
            ##### zeroth 실행 후 lat 파일 추출 => lat 파일로 phone list get 
            ##### 해당 phone list를 txt로 저장 후, 여기로 불러오기

        result = np.array(sp_mat_list)
        tar_phone = None if phone_list == None else np.array(phone_list)
        return result, tar_phone
    
    else :
        result = np.array(sp_mat_list)
        tar_phone = None
        return result, tar_phone

def sample_train_data5(sp_list, n_frames=128, shuffle=False, ppg_list=None, phone_list=None, voted=False):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """
    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    t_phone_list = []

    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_phone = phone_list[idx]

        cur_data_frames = cur_data.shape[-1]
        #print("data len : {}, phone len : {}".format(cur_data_frames, len(cur_phone)))

        assert cur_data_frames >= n_frames, "Too short SP"
        #assert len(cur_phone) == (cur_data_frames//4), "Phone length is wrong {} {}".format(cur_data_frames, len(cur_phone))
        if len(cur_phone) < (cur_data_frames//4) :
                left = (cur_data_frames//4) - len(cur_phone)
                last = cur_phone[-1]
                for i in range(left) :
                    cur_phone.append(last)
        elif len(cur_phone) > (cur_data_frames//4) : 
            left = len(cur_phone) - (cur_data_frames//4)
            cur_phone = cur_phone[:(cur_data_frames//4)]
        else :
            left = 0

        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        # start_idx = np.random.randint(cur_data_frames - n_frames)
        end_idx = start_idx + n_frames

        #print("start:{}, {}".format(start_idx, int(start_idx / 4)))
        #print("end : {}, {}".format(end_idx, int(start_idx / 4) + int(n_frames / 4)))

        cur_sp_mat = cur_data[:, start_idx:end_idx]
        target_phone = cur_phone[start_idx//4 : end_idx//4]
        #print(cur_phone)
        #print(target_phone)
    
        assert len(target_phone) == 32, "phone len is wrong ... {}".format(len(target_phone))

        sp_mat_list.append(cur_sp_mat)
        t_phone_list.append(target_phone)

    #print("-----------------Final Data Set has {} data".format(len(t_phone_list)))
    result = np.array(sp_mat_list)
    tar_phone = None if phone_list == None else np.array(t_phone_list)
    return result, tar_phone

def sample_train_data_1202(sp_list, n_frames=128, shuffle=False, ppg_list=None, phone_list=None, voted=False):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """
    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    t_phone_list = []

    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_phone = phone_list[idx]

        cur_data_frames = cur_data.shape[-1]
        #print("data len : {}, phone len : {}".format(cur_data_frames, len(cur_phone)))

        assert cur_data_frames >= n_frames, "Too short SP"
        #assert len(cur_phone) == (cur_data_frames//4), "Phone length is wrong {} {}".format(cur_data_frames, len(cur_phone))
        if len(cur_phone) < (cur_data_frames//8) :
                left = (cur_data_frames//8) - len(cur_phone)
                last = cur_phone[-1]
                for i in range(left) :
                    cur_phone.append(last)
        elif len(cur_phone) > (cur_data_frames//8) : 
            left = len(cur_phone) - (cur_data_frames//8)
            cur_phone = cur_phone[:(cur_data_frames//8)]
        else :
            left = 0

        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        # start_idx = np.random.randint(cur_data_frames - n_frames)
        end_idx = start_idx + n_frames

        #print("start:{}, {}".format(start_idx, int(start_idx / 4)))
        #print("end : {}, {}".format(end_idx, int(start_idx / 4) + int(n_frames / 4)))

        cur_sp_mat = cur_data[:, start_idx:end_idx]
        target_phone = cur_phone[start_idx//8 : (start_idx//8)+16]
        #print(cur_phone)
        #print(target_phone)
    
        assert len(target_phone) == 16, "phone len is wrong ... {}".format(len(target_phone))

        sp_mat_list.append(cur_sp_mat)
        t_phone_list.append(target_phone)

    #print("-----------------Final Data Set has {} data".format(len(t_phone_list)))
    result = np.array(sp_mat_list)
    tar_phone = None if phone_list == None else np.array(t_phone_list)
    return result, tar_phone

def multiple_sample_train_data(sp_list, n_frames=128, shuffle=False, ppg_list=None):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """

    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    target_list = []

    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_data_frames = cur_data.shape[-1]
        
        assert cur_data_frames >= n_frames, "Too short SP"

        cur_sp_mat = []
        cur_target = []
        for seg_idx in range(0,cur_data_frames,n_frames):
            start_idx = seg_idx
            end_idx = seg_idx+n_frames

            if end_idx >= cur_data_frames:
                break

            cur_sp_mat.append(cur_data[:, start_idx:end_idx])

            if ppg_list is not None:
                cur_ppg = ppg_list[idx]
                cur_ppg_mat = cur_ppg[:, start_idx:end_idx]
                cur_target.append(extract_target_from_ppg(cur_ppg_mat, window=4))
                

        sp_mat_list.extend(cur_sp_mat)
        if ppg_list is not None:
            target_list.extend(cur_target)

    result = np.array(sp_mat_list)
    targets = None if ppg_list == None else np.array(target_list)
    return result, targets

def feat_loader_MD(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None):
    """
    spk_labs: int
    """
    total_feat = []
    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        spk_feats = []
        spk_targets = []
        sampled_sp, targets = sample_train_data(sp_list, n_frames=n_frames, ppg_list=ppg_list)
        
        # crawl wav sample
        for cur_sp in sampled_sp:
            spk_feats.append(cur_sp)
        
        if ppg_dict != None:
            for cur_target in targets:
                spk_targets.append(cur_target)

        # shuffle and pack        
        if shuffle:
            np.random.shuffle(spk_feats)
        feat_num = len(spk_feats)
        for start_idx in range(0, feat_num, batch_size):
            end_idx = start_idx + batch_size
            sps = spk_feats[start_idx:end_idx]
            labs = None if ppg_dict == None else spk_targets[start_idx:end_idx]
            
            total_feat.append((spk_idx, sps, labs))
    
    total_num = len(total_feat)
    if shuffle:
        np.random.shuffle(total_feat)
    
    for cur_idx in range(0, total_num):
        spk_idx, sps, labs = total_feat[cur_idx]

        x = np.expand_dims(sps, axis=1)
        x = torch.Tensor(x).float().cuda()

        if labs != None:
            t = torch.Tensor(labs).long().cuda()
            x = (x, t)

        yield x, spk_idx

def feat_loader_single(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None, is_AC=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_ppgs = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        sampled_sp, targets = sample_train_data(sp_list, n_frames=n_frames, ppg_list=ppg_list)
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )
        if ppg_list is not None:
            for target in targets:
                total_ppgs.append(target)
    
    
    total_num = len(total_feat)
    total_idxs = np.arange(total_num)

    if shuffle:
        np.random.shuffle(total_idxs)

    if is_AC:
        np.random.shuffle(total_idxs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        index_list = total_idxs[start_idx:end_idx]

        spk_idxs = []

        x=[]
        t=[]

        for cur_idx in index_list:
            spk_idx, sp = total_feat[cur_idx]

            spk_idxs.append(spk_idx)
            x.append(sp)
            
            if ppg_dict is not None:
                t.append(total_ppgs[cur_idx])

        # post processing
        x = np.expand_dims(x, axis=1)
        x = torch.Tensor(x).float().cuda()

        if ppg_dict is not None:
            t = torch.Tensor(t).long().cuda()
            x = (x, t)
        else :
            x = (x, None)
        
        yield spk_idxs, x

def feat_loader_single2(stat_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None, is_AC=False, voted=False, is_dev=False, is_phone=True):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(stat_dict)
    total_feat = []
    total_ppgs = []
    total_phones = []

    for spk_idx, (spk_id, t) in enumerate(stat_dict.items()):

        if is_dev : 
            sp_list = t[1]
            sps = t[0]
            f0s = t[2]
            aps = t[3]
        else :
            sp_list = t[0]
            f0s = t[5]
            sps = t[6]
            aps = t[7]

        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        # kaldi_1 data
        # sampled_sp, targets, src_phone = sample_train_data2(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=voted)
        # sciprt->phone data
        sampled_sp, tar_phones = sample_train_data4(sp_list, n_frames=n_frames, ppg_list=ppg_list, f0s=f0s, sps=sps, aps=aps, voted=voted, is_phone=is_phone)
        # 128프레임씩 잘린 것들을 다시 total_feat 라는 list에 append 한다
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )
        
        if is_phone :
            for tar_phone in tar_phones :
                total_phones.append(
                    (spk_idx, tar_phone)
                )
    
    total_num = len(total_feat)
    total_idxs = np.arange(total_num)

    if shuffle:
        np.random.shuffle(total_idxs)

    if is_AC:
        np.random.shuffle(total_idxs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        index_list = total_idxs[start_idx:end_idx]

        spk_idxs = []
        spk_names = []

        x=[]
        t_phone = []

        for cur_idx in index_list:
            spk_idx, sp = total_feat[cur_idx]
            spk_idxs.append(spk_idx)
            spk_names.append(spk_id)
            x.append(sp)
        if is_phone :
            for cur_idx in index_list:
                _, p = total_phones[cur_idx]
                t_phone.append(p)

        #print("x shape : {}".format(np.array(x).shape))
        # post processing
        # x는 batch개의 (36, 128) 차원을 가지는 데이터 (128데이터)
        x = np.expand_dims(x, axis=1)
        # x.shape = (36, 1, 128)
        #print("preprocessed x shape : {}".format(np.array(x).shape))
        #print("preprocessed phone shape : {}".format(np.array(t_phone).shape))
        x = torch.Tensor(x).float().cuda()
        if is_phone :
            t_phone = torch.Tensor(t_phone).long().cuda()    
        else :
            t_phone = None

        #yield spk_idxs, x
        yield spk_idxs, spk_names, x, t_phone
        
def feat_loader_single_1006(sp_dict, batch_size, n_frames=128, phone_dict=None, shuffle=False, ppg_dict=None, is_AC=False, voted=False, is_dev=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_phones = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):

        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        phone_list = None if phone_dict == None else phone_dict[spk_id]
        sampled_sp, _, targets = sample_train_data3(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=False)
    
        # kaldi_1 data
        # sampled_sp, targets, src_phone = sample_train_data2(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=voted)
        # sciprt->phone data
        # 128프레임씩 잘린 것들을 다시 total_feat 라는 list에 append 한다
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )
        if phone_list is not None :
            for t_phone in targets :
                total_phones.append(t_phone)
    
    total_num = len(total_feat)
    total_idxs = np.arange(total_num)

    if shuffle:
        np.random.shuffle(total_idxs)

    if is_AC:
        np.random.shuffle(total_idxs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        index_list = total_idxs[start_idx:end_idx]

        spk_idxs = []
        spk_names = []

        x=[]
        t_phone = []

        for cur_idx in index_list:
            spk_idx, sp = total_feat[cur_idx]
            spk_idxs.append(spk_idx)
            spk_names.append(spk_id)
            x.append(sp)
            if phone_dict is not None : 
                t_phone.append(total_phones[cur_idx])

        #print("x shape : {}".format(np.array(x).shape))
        # post processing
        # x는 batch개의 (36, 128) 차원을 가지는 데이터 (128데이터)
        x = np.expand_dims(x, axis=1)
        # x.shape = (36, 1, 128)
        #print("preprocessed x shape : {}".format(np.array(x).shape))
        x = torch.Tensor(x).float().cuda()

        if phone_dict is not None :
            t_phone = torch.Tensor(t_phone).long().cuda()   
            x = (x, t_phone) 

        #yield spk_idxs, x
        yield spk_idxs, x

def feat_loader_single3(sp_dict, batch_size, n_frames=128, phone_dict=None, shuffle=False, ppg_dict=None, is_AC=False, voted=False, is_dev=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_phones = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):

        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        phone_list = None if phone_dict == None else phone_dict[spk_id]
        sampled_sp, targets = sample_train_data5(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=False)
    
        # kaldi_1 data
        # sampled_sp, targets, src_phone = sample_train_data2(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=voted)
        # sciprt->phone data
        # 128프레임씩 잘린 것들을 다시 total_feat 라는 list에 append 한다
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )
        if phone_list is not None :
            for t_phone in targets :
                total_phones.append(t_phone)
    
    total_num = len(total_feat)
    total_idxs = np.arange(total_num)

    if shuffle:
        np.random.shuffle(total_idxs)

    if is_AC:
        np.random.shuffle(total_idxs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        index_list = total_idxs[start_idx:end_idx]

        spk_idxs = []
        spk_names = []

        x=[]
        t_phone = []

        for cur_idx in index_list:
            spk_idx, sp = total_feat[cur_idx]
            spk_idxs.append(spk_idx)
            spk_names.append(spk_id)
            x.append(sp)
            if phone_dict is not None : 
                t_phone.append(total_phones[cur_idx])

        #print("x shape : {}".format(np.array(x).shape))
        # post processing
        # x는 batch개의 (36, 128) 차원을 가지는 데이터 (128데이터)
        x = np.expand_dims(x, axis=1)
        # x.shape = (36, 1, 128)
        #print("preprocessed x shape : {}".format(np.array(x).shape))
        x = torch.Tensor(x).float().cuda()

        if phone_dict is not None :
            t_phone = torch.Tensor(t_phone).long().cuda()   
            x = (x, t_phone) 

        #yield spk_idxs, x
        yield spk_idxs, x

def feat_loader_single_1202(sp_dict, batch_size, n_frames=128, phone_dict=None, shuffle=False, ppg_dict=None, is_AC=False, voted=False, is_dev=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_phones = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):

        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        phone_list = None if phone_dict == None else phone_dict[spk_id]
        sampled_sp, targets = sample_train_data_1202(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=False)
    
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )
        if phone_list is not None :
            for t_phone in targets :
                total_phones.append(t_phone)
    
    total_num = len(total_feat)
    total_idxs = np.arange(total_num)

    if shuffle:
        np.random.shuffle(total_idxs)

    if is_AC:
        np.random.shuffle(total_idxs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        index_list = total_idxs[start_idx:end_idx]

        spk_idxs = []
        spk_names = []

        x=[]
        t_phone = []

        for cur_idx in index_list:
            spk_idx, sp = total_feat[cur_idx]
            spk_idxs.append(spk_idx)
            spk_names.append(spk_id)
            x.append(sp)
            if phone_dict is not None : 
                t_phone.append(total_phones[cur_idx])

        #print("x shape : {}".format(np.array(x).shape))
        # post processing
        # x는 batch개의 (36, 128) 차원을 가지는 데이터 (128데이터)
        x = np.expand_dims(x, axis=1)
        # x.shape = (36, 1, 128)
        #print("preprocessed x shape : {}".format(np.array(x).shape))
        x = torch.Tensor(x).float().cuda()

        if phone_dict is not None :
            t_phone = torch.Tensor(t_phone).long().cuda()   
            x = (x, t_phone) 

        #yield spk_idxs, x
        yield spk_idxs, x

def feat_loader_multiple(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None, is_AC=False):
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_ppgs = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        sampled_sp, targets = multiple_sample_train_data(sp_list, n_frames=n_frames, ppg_list=ppg_list)

        spk_sp_idxs_num = len(sampled_sp)
        spk_sp_idxs = np.arange(spk_sp_idxs_num)

        np.random.shuffle(spk_sp_idxs)

        # minimum 291
        spk_sp_idxs = spk_sp_idxs[:200]
        for i in spk_sp_idxs:
            total_feat.append(
                (spk_idx, sampled_sp[i])
            )

            if ppg_list is not None:
                total_ppgs.append(targets[i])
    
    
    total_num = len(total_feat)
    sp_per_spk = total_num//TOTAL_SPK_NUM

    total_idxs = np.arange(total_num)
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])


    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)


    total_pairs = []
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)

    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            if ppg_dict is not None:
                src_t.append(total_ppgs[src])

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)
        
                if ppg_dict is not None:
                    tar_t_pt.append(total_ppgs[tar])
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            tar_t.append(tar_t_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()
        # print(src_x.shape)


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)
        # print(tar_x.shape)

        if ppg_dict is not None:
            src_t = torch.Tensor(src_t).long().cuda()
            src_x = (src_x, src_t)

            # tar_t = torch.Tensor(tar_t).long().cuda()
            # tar_t = tar_t.permute(2,0,1,3,4)

            # tar_x = (tar_x, tar_t)
        
        yield src_idxs, src_x, tar_idxs, tar_x

def feat_loader_pair(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None,is_MD=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_ppgs = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        sampled_sp, targets = sample_train_data(sp_list, n_frames=n_frames, ppg_list=ppg_list)
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )
        if ppg_list is not None:
            for target in targets:
                total_ppgs.append(target)
    
    
    total_num = len(total_feat)
    sp_per_spk = total_num//TOTAL_SPK_NUM

    total_idxs = np.arange(total_num)
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])


    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)

    total_pairs = []
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)

    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            if ppg_dict is not None:
                src_t.append(total_ppgs[src])

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)
        
                if ppg_dict is not None:
                    tar_t_pt.append(total_ppgs[tar])
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            tar_t.append(tar_t_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)

        if ppg_dict is not None:
            src_t = torch.Tensor(src_t).long().cuda()
            src_x = (src_x, src_t)

            # tar_t = torch.Tensor(tar_t).long().cuda()
            # tar_t = tar_t.permute(2,0,1,3,4)

            # tar_x = (tar_x, tar_t)
        else :
            src_x = (src_x, None)
        
        yield src_idxs, src_x, tar_idxs, tar_x
# 
def feat_loader_pair2(stat_dict, batch_size, n_frames=128, shuffle=False, phone_dict = None, ppg_dict=None, is_phone=True, is_MD=False, voted=False, is_dev=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(stat_dict)
    total_feat = []
    total_ppgs = []
    total_phones = []

    for spk_idx, (spk_id, t) in enumerate(stat_dict.items()):
        if is_dev : 
            sp_list = t[1]
            sps = t[0]
            f0s = t[2]
            aps = t[3]
        else :
            sp_list = t[0]
            f0s = t[5]
            sps = t[6]
            aps = t[7]

        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        # kaldi_1 data
        # sampled_sp, targets, src_phone = sample_train_data2(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=voted)
        # sciprt->phone data
        sampled_sp, tar_phones = sample_train_data4(sp_list, n_frames=n_frames, ppg_list=ppg_list, f0s=f0s, sps=sps, aps=aps, voted=voted, is_phone=is_phone)
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )

        '''
        if ppg_list is not None:
            for target in targets:
                total_ppgs.append(target)
        '''

        if is_phone : 
            for tar_phone in tar_phones :
                total_phones.append(
                    (spk_idx, tar_phone)
                )

    total_num = len(total_feat) #1600 581
    total_num = (total_num//10) * 10
    total_feat = total_feat[:total_num]
    sp_per_spk = total_num//TOTAL_SPK_NUM #160 58.1

    total_idxs = np.arange(total_num) # 1280 500
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = [] #8
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])

    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)

    total_pairs = []
    print("-----------------------------------")
    #print(TOTAL_SPK_NUM)
    #print(len(total_idxs))
    #print(len(tar_sp_idx))
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)

    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]
        src_phone=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]
        tar_phone=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            '''
            if ppg_dict is not None:
                src_t.append(total_ppgs[src])
            '''

            if is_phone :
                _, src_p = total_phones[src]
                src_phone.append(src_p)

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []
            tar_phone_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)
        
                '''
                if ppg_dict is not None:
                    tar_t_pt.append(total_ppgs[tar])
                '''

                if is_phone :
                    _, tar_p = total_phones[src]
                    tar_phone_pt.append(total_phones[tar_p])
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            # tar_t.append(tar_t_pt)
            tar_phone.append(tar_phone_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)

        if ppg_dict is not None:
            # src_t = torch.Tensor(src_t).long().cuda()
            #src_x = (src_x, src_t)
            src_x = (src_x, None)
            # tar_t = torch.Tensor(tar_t).long().cuda()
            # tar_t = tar_t.permute(2,0,1,3,4)

            # tar_x = (tar_x, tar_t)
        
        if is_phone :
            src_phone = torch.Tensor(src_phone).long().cuda()
        else:
            src_phone = None

        yield src_idxs, src_x, src_phone, tar_idxs, tar_x

def feat_loader_pair_1006(sp_dict, batch_size, n_frames=128, shuffle=False, phone_dict=None, ppg_dict=None, is_MD=False, is_dev=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_ppgs = []
    total_phones = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        phone_list = None if phone_dict == None else phone_dict[spk_id]
        sampled_sp, _, tar_phones = sample_train_data3(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=False)
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )

        '''
        if ppg_list is not None:
            for target in targets:
                total_ppgs.append(target)
        '''

        if phone_list is not None : 
            for tar_phone in tar_phones :
                total_phones.append(
                    (spk_idx, tar_phone)
                )

    total_num = len(total_feat) #1600 581
    total_num = (total_num//10) * 10
    total_feat = total_feat[:total_num]
    sp_per_spk = total_num//TOTAL_SPK_NUM #160 58.1

    total_idxs = np.arange(total_num) # 1280 500
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = [] #8
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])

    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)

    total_pairs = []
    print("-----------------------------------")
    #print(TOTAL_SPK_NUM)
    #print(len(total_idxs))
    #print(len(tar_sp_idx))
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)

    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]
        src_phone=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]
        tar_phone=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            '''
            if ppg_dict is not None:
                src_t.append(total_ppgs[src])
            '''

            if phone_list is not None :
                _, src_p = total_phones[src]
                src_phone.append(src_p)

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []
            tar_phone_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)
        
                '''
                if ppg_dict is not None:
                    tar_t_pt.append(total_ppgs[tar])
                '''

                if phone_list is not None :
                    _, tar_p = total_phones[tar]
                    tar_phone_pt.append(tar_p)
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            # tar_t.append(tar_t_pt)
            tar_phone.append(tar_phone_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)

        if ppg_dict is not None:
            src_t = torch.Tensor(src_t).long().cuda()
            src_x = (src_x, src_t)
            tar_t = torch.Tensor(tar_t).long().cuda()
            tar_t = tar_t.permute(2,0,1,3,4)
            tar_x = (tar_x, tar_t)
        else :
            src_x = (src_x, None)
        
        if phone_list is not None :
            src_phone = torch.Tensor(src_phone).long().cuda()
        else:
            src_phone = None

        yield src_idxs, src_x, src_phone, tar_idxs, tar_x

def feat_loader_pair_1202(sp_dict, batch_size, n_frames=128, shuffle=False, phone_dict=None, ppg_dict=None, is_MD=False, is_dev=False):
    """
    spk_labs: (VAE3) list of ints [int, int, int, ...]  // (MD) int 
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_ppgs = []
    total_phones = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        phone_list = None if phone_dict == None else phone_dict[spk_id]
        sampled_sp, tar_phones = sample_train_data_1202(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list, voted=False)
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )

        if phone_list is not None : 
            for tar_phone in tar_phones :
                total_phones.append(
                    (spk_idx, tar_phone)
                )

    total_num = len(total_feat) #1600 581
    #total_num = (total_num//10) * 10
    total_feat = total_feat[:total_num]
    sp_per_spk = total_num//TOTAL_SPK_NUM #160 58.1

    total_idxs = np.arange(total_num) # 1280 500
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = [] #8
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])

    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)

    total_pairs = []
    print("-----------------------------------")
    #print(TOTAL_SPK_NUM)
    #print(len(total_idxs))
    #print(len(tar_sp_idx))
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]
        src_phone=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]
        tar_phone=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            if phone_list is not None :
                _, src_p = total_phones[src]
                src_phone.append(src_p)

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []
            tar_phone_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)

                if phone_list is not None :
                    _, tar_p = total_phones[tar]
                    tar_phone_pt.append(tar_p)
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            # tar_t.append(tar_t_pt)
            tar_phone.append(tar_phone_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)

        if ppg_dict is not None:
            src_t = torch.Tensor(src_t).long().cuda()
            src_x = (src_x, src_t)
            tar_t = torch.Tensor(tar_t).long().cuda()
            tar_t = tar_t.permute(2,0,1,3,4)
            tar_x = (tar_x, tar_t)
        else :
            src_x = (src_x, None)
        
        if phone_list is not None :
            src_phone = torch.Tensor(src_phone).long().cuda()
        else:
            src_phone = None

        yield src_idxs, src_x, src_phone, tar_idxs, tar_x

def feat_loader_pair3(sp_dict, batch_size, n_frames=128, shuffle=False, phone_dict=None, ppg_dict=None, is_MD=False, is_dev=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_ppgs = []
    total_phones = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        phone_list = None if phone_dict == None else phone_dict[spk_id]
        sampled_sp, tar_phones = sample_train_data5(sp_list, n_frames=n_frames, ppg_list=ppg_list, phone_list=phone_list)
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )

        '''
        if ppg_list is not None:
            for target in targets:
                total_ppgs.append(target)
        '''

        if phone_list is not None : 
            for tar_phone in tar_phones :
                total_phones.append(
                    (spk_idx, tar_phone)
                )

    total_num = len(total_feat) #1600 581
    total_num = (total_num//10) * 10
    total_feat = total_feat[:total_num]
    sp_per_spk = total_num//TOTAL_SPK_NUM #160 58.1

    total_idxs = np.arange(total_num) # 1280 500
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = [] #8
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])

    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)

    total_pairs = []
    print("-----------------------------------")
    #print(TOTAL_SPK_NUM)
    #print(len(total_idxs))
    #print(len(tar_sp_idx))
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)

    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]
        src_phone=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]
        tar_phone=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            '''
            if ppg_dict is not None:
                src_t.append(total_ppgs[src])
            '''

            if phone_list is not None :
                _, src_p = total_phones[src]
                src_phone.append(src_p)

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []
            tar_phone_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)
        
                '''
                if ppg_dict is not None:
                    tar_t_pt.append(total_ppgs[tar])
                '''

                if phone_list is not None :
                    _, tar_p = total_phones[tar]
                    tar_phone_pt.append(tar_p)
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            # tar_t.append(tar_t_pt)
            tar_phone.append(tar_phone_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)

        if ppg_dict is not None:
            src_t = torch.Tensor(src_t).long().cuda()
            src_x = (src_x, src_t)
            tar_t = torch.Tensor(tar_t).long().cuda()
            tar_t = tar_t.permute(2,0,1,3,4)
            tar_x = (tar_x, tar_t)
        else :
            src_x = (src_x, None)
        
        if phone_list is not None :
            src_phone = torch.Tensor(src_phone).long().cuda()
        else:
            src_phone = None
        #print(src_x[0].shape, src_phone.shape)
        yield src_idxs, src_x, src_phone, tar_idxs, tar_x

def get_loader(SP_DICT, batch_size, n_frames=128, shuffle=False, PPG_DICT=None, is_MD=False, is_AC=False):
    data_loader = None
    if is_MD:
        data_loader = feat_loader_MD(SP_DICT, batch_size, n_frames=n_frames, shuffle=shuffle, ppg_dict=PPG_DICT)
    else:
        data_loader = feat_loader_single(SP_DICT, batch_size, n_frames=n_frames, shuffle=shuffle, ppg_dict=PPG_DICT, is_AC=is_AC)

    return data_loader

########################
"""
MD: spk_idx => A_y (batch_len, vec_dim)
VAE: spk_idxs => A_y 

spk_idx => make_spk_vector, make_spk_target
"""
########################
def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec

def expand_spk_vec(spk_vec, batch_len):
    spk_vec = np.expand_dims(spk_vec, axis=0)
    y = np.repeat(spk_vec, batch_len, axis=0)
    y = torch.Tensor(y).float().cuda()
    return y

def make_spk_vector(spk_idxs, spk_num, batch_len=0, is_MD=False):
    A_y = []
    if is_MD:
        spk_idx = spk_idxs
        spk_vec = make_one_hot_vector(spk_idx, spk_num)
        A_y = expand_spk_vec(spk_vec, batch_len)
    else:
        for spk_idx in spk_idxs:
            spk_vec = make_one_hot_vector(spk_idx, spk_num)
            A_y.append(spk_vec)
        A_y = np.array(A_y)
        A_y = torch.Tensor(A_y).float().cuda()
    return A_y

########################
def make_lab(spk_idx, batch_len):
    t = torch.Tensor([spk_idx]).long().cuda()
    t = t.repeat((batch_len))
    return t

def make_spk_target(spk_idxs, batch_len=0, is_MD=False):
    A_spk_lab = []
    if is_MD:
        spk_idx = spk_idxs
        A_spk_lab = make_lab(spk_idx, batch_len)
    else:
        A_spk_lab = [spk_idx for spk_idx in spk_idxs]
        A_spk_lab = torch.Tensor(A_spk_lab).long().cuda()

    return A_spk_lab

########################

def get_all_target_idx(A_spk_idxs, spk_num, is_MD=False):
    """
    A_spk_idxs: [int, int, int, ...] or int
    B_spk_idxs: [[int, int, int], [int, int, int], ...] or [int, int, int]
    """
    result=[]

    if is_MD:
        B_spk_idx_list = []
        src_spk_idx = A_spk_idxs
        for trg_spk_idx in range(spk_num):
            # skip when source speaker is same with target speaker 
            if src_spk_idx == trg_spk_idx:
                continue
            B_spk_idx_list.append(trg_spk_idx)
        result = np.array(B_spk_idx_list)
    else:
        for src_spk_idx in A_spk_idxs:
            B_spk_idx_list = []
            for trg_spk_idx in range(spk_num):
                # skip when source speaker is same with target speaker 
                if src_spk_idx == trg_spk_idx:
                    continue
                B_spk_idx_list.append(trg_spk_idx) # (3)
            result.append(B_spk_idx_list) # (batch_len, 3)
        result = np.swapaxes(np.array(result), 0, 1) # (3, batch_len)

    return result