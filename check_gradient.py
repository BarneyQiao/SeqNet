# -*- coding: utf-8 -*-
"""
@Time          : 2021/12/15 16:13
@Author        : BarneyQ
@File          : check_gradient.py
@Software      : PyCharm
@Description   :
@Modification  :
    @Author    :
    @Time      :
    @Detail    :

"""
from torch.autograd import gradcheck
import torch
import torch.nn as nn

inputs = torch.randn((10, 5), requires_grad=True, dtype=torch.double)
# linear = nn.Linear(5, 3)
linear = inputs.split([1,9],0)
# linear = linear.double()
test = gradcheck(lambda x: linear(x), inputs)
print("Are the gradients correct: ", test)