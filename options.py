#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import backbones
import heads
import losses

backbone_names = sorted(name for name in backbones.__dict__
if not name.startswith("__") and callable(backbones.__dict__[name]))

head_names = sorted(name for name in heads.__dict__
if not name.startswith("__") and callable(heads.__dict__[name]))

loss_names = sorted(name for name in losses.__dict__
if not name.startswith("__") and callable(losses.__dict__[name]))
optimizer_names = ['Adam','SGD']

def get_options():
    parser = argparse.ArgumentParser(description='Classification framework in PyTorch')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to experiment folder (default: none), all arguments loaded from the folder')
    
    parser.add_argument('--backbone', default='resnet50',
                        choices=backbone_names,
                        help='backbone architectures: ' + ' | '.join(backbone_names))
    parser.add_argument('--head', default='DC_EPCC',
                        choices=head_names,
                        help='head functions: ' + ' | '.join(head_names))
    parser.add_argument('--loss', default='hinge_mc_v2',
                        choices=loss_names,
                        help='loss functions: ' + ' | '.join(loss_names))
    
    parser.add_argument('--kapa', default=1.0, type=float, metavar='compactness term',
                        help='cost penalty for compactness')
    parser.add_argument('--margin', default=1.0, type=float, metavar='margin for hinge loss',
                        help='margin')
    
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    
    parser.add_argument('--optimizer', default='SGD',
                            choices=optimizer_names,
                            help='optimizers: ' + ' | '.join(optimizer_names))
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    
    parser.add_argument('--onevsrest', action='store_true',default=False)
    parser.add_argument('--centerloss', action='store_true',default=False)
    parser.add_argument('--vis2d', action='store_true',default=False)
    parser.add_argument('--parse-epcc-params', action='store_true',default=False)
    args = parser.parse_args()
    return args