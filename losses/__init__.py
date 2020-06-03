#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from losses.center_reg import *
from losses.hinge_loss import *
# import torch loss functions
import torch
submodules = [key for key in torch.nn.__dict__.keys() if 'Loss' in key]
for sublibrary in submodules:
    try:
        exec("from {m} import {s}".format(m='torch.nn', s=sublibrary))
    except Exception as e:
        print(e)