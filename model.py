#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: model.py 
@time: 2019/07/09
@contact: wchao118@gmail.com
@software: PyCharm 
"""

import torch
import os
import torch.nn as nn
nn.utils.rnn.pad_packed_sequence()


class ASR(nn.Module):
    def __init__(self, batch_size, seq_len, feature_dim=10, num_hiddens=128, num_classes=27):
        super(ASR, self).__init__()

        self.num_classes = num_classes
        self.num_hiddens = num_hiddens
        self.batch_size = batch_size
        self.seq_len = seq_len

        # input_size:The number of expected features in the input `x`
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=num_hiddens, batch_first=True, num_layers=2)
        self.cls = nn.Linear(seq_len*num_hiddens,  seq_len*num_classes)

    def forward(self, input):
        out, _ = self.lstm(input)
        out = out.reshape(self.batch_size, -1)
        cls = self.cls(out)
        cls = cls.reshape(self.batch_size, -1, self.num_classes)
        return cls


if __name__ == '__main__':
    # batch_size, sequence_len(maybe vary from input), features
    x = torch.ones((4, 5, 10))
    asrmode = ASR(batch_size=4, seq_len=5)
    asrmode(x)
