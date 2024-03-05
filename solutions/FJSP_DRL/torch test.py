# -*- coding: utf-8 -*-
# @Time    : 2024/2/22 17:26
# @Author  : Hongxinag Zhang
# @Email   : hongxiang@my.swjtu.edu.cn
# @File    : torch test.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

x = torch.ones(20,6,50)

x[10,5,34] = 3

print(x[10,5,34], x[10,5,34].shape)

