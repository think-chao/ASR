#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: config.py
@time: 2019/07/08
@contact: wchao118@gmail.com
@software: PyCharm 
"""

data_root = './data'
batch_size = 1
lr = 0.001
epoch = 50

cur_id = 0

space = '<space>'
space_index = 0

first_index = ord('a')-1

num_classes = ord('z') - ord('a') + 1 + 1 + 1  # 26char + blank or space

