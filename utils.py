#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: utils.py 
@time: 2019/07/08
@contact: wchao118@gmail.com
@software: PyCharm 
"""

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import random
import numpy as np
import os
import config

all_audio = [file for file in os.listdir(config.data_root) if file.split('.')[-1] == 'wav']
all_txt = [file for file in os.listdir(config.data_root) if file.split('.')[-1] == 'txt']
audio_len = len(all_audio)


def gen_next_batch(cur_id=0, batch_size=1):
    if cur_id == 0:
        random.shuffle(all_audio)
    if audio_len - cur_id / batch_size < 1:
        cur_id = 0
    batch_audio_name = all_audio[cur_id:cur_id + batch_size]

    inputs = []
    for file in batch_audio_name:
        fs, audio = wav.read(os.path.join(config.data_root, file))
        mfcc_feature = mfcc(audio)
        label = get_target(os.path.join(config.data_root, file.split('.')[0] + '.txt'))
        inputs.append(mfcc_feature)
    inputs = np.array(inputs)


def get_target(file_name):
    with open(file_name, 'r') as f:
        line = f.readlines()[-1]
        original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
        print(original)
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')
        targets = np.hstack([config.space if x == '' else list(x) for x in targets])
        targets = np.asarray([config.space_index if x == config.space else ord(x) - config.first_index
                              for x in targets])
        print(targets)
    os._exit(0)

    return _


def sparse_tuple_from(sequences, dtype=np.int32):
    pass


def test():
    import torch
    from torch.autograd import Variable
    from warpctc_pytorch import CTCLoss
    ctc_loss = CTCLoss()
    # expected shape of seqLength x batchSize x alphabet_size
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
    labels = Variable(torch.IntTensor([1, 2]))
    label_sizes = Variable(torch.IntTensor([2]))
    probs_sizes = Variable(torch.IntTensor([2]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    cost.backward()
    print('PyTorch bindings for Warp-ctc')


gen_next_batch()
