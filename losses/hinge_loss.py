#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

__all__ = ['hinge','hinge_inverted',
           'hinge_mc_v2','hinge_mc_v3',
           'hinge_onevsrest_v0','hinge_onevsrest_v1']

class hinge(nn.Module):
    def __init__(self,margin=1.0):
        super(hinge,self).__init__()
        self.margin = margin
    def forward(self,input1,target):
        loss = (-target * input1.t() + self.margin).clamp(min=0)
        # loss = torch.pow(loss,2) 
        # loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
        return loss.mean()

class hinge_inverted(nn.Module):
    def __init__(self,margin=1.0):
        super(hinge_inverted,self).__init__()
        self.margin = margin
    def forward(self,input1,target):
        target_mod = torch.from_numpy(np.where(target.cpu().numpy()==1.0,-1.0,1.0)).cuda().float()
        loss = (-target_mod * input1.t() + self.margin).clamp(min=0)
        # loss = torch.pow(loss,2) 
        # loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
        return loss.mean()

class hinge_mc_v2(nn.Module):
    def __init__(self,margin,interclass_loss=True):
        super(hinge_mc_v2,self).__init__()
        self.margin = margin
        self.interclass_loss = interclass_loss
    def forward(self,outputs,labels):
        batch_size,num_classifier = outputs.shape
        classes = torch.arange(num_classifier).long().cuda()
        labels2mask = labels.unsqueeze(1).expand(batch_size, num_classifier)
        mask = labels2mask.eq(classes.expand(batch_size, num_classifier))
        concat_labels = torch.from_numpy(np.where(mask.cpu(),1,-1)).cuda()
        
        bin_loss = torch.clamp(self.margin - outputs.t()*concat_labels.t(), min=0)
        intcls_loss = torch.zeros(num_classifier).cuda()
        if(self.interclass_loss):
            for i in range(num_classifier):
                if(outputs[labels==i].shape[0] != 0):
                    mask = np.arange(num_classifier)!=i
                    intcls_loss[i] = torch.mean(torch.clamp(self.margin-(outputs[labels==i][:,i].unsqueeze(1) - outputs[labels==i][:,mask]),min=0))
        return bin_loss.mean() + intcls_loss.mean()

class hinge_mc_v3(nn.Module):
    def __init__(self,margin,interclass_loss=True):
        super(hinge_mc_v3,self).__init__()
        self.margin = margin
        self.interclass_loss = interclass_loss
        self.interclass_criterion = nn.MultiMarginLoss(margin=margin)
    def forward(self,outputs,labels):
        batch_size,num_classifier = outputs.shape
        classes = torch.arange(num_classifier).long().cuda()
        labels2mask = labels.unsqueeze(1).expand(batch_size, num_classifier)
        mask = labels2mask.eq(classes.expand(batch_size, num_classifier))
        concat_labels = torch.from_numpy(np.where(mask.cpu(),1,-1)).cuda()
        bin_loss = torch.clamp(self.margin - outputs.t()*concat_labels.t(), min=0)
        if(self.interclass_loss):
            intcls_loss = self.interclass_criterion(outputs,labels)
        return bin_loss.mean() + intcls_loss

class hinge_onevsrest_v0(nn.Module):
    def __init__(self,margin,interclass_loss=False):
        super(hinge_onevsrest_v0,self).__init__()
        self.margin = margin
    def forward(self,outputs,labels):
        num_classifier = outputs.shape[-1]
        for i in range(num_classifier):
            if i==0:
                loss = torch.mean(torch.clamp(self.margin - outputs[:,i].t() * torch.from_numpy(np.where(labels[:,i].cpu().numpy()>=0,1.0,-1.0)).cuda().long(), min=0))
            else:
                loss = loss + torch.mean(torch.clamp(self.margin - outputs[:,i].t() * torch.from_numpy(np.where(labels[:,i].cpu().numpy()>=0,1.0,-1.0)).cuda().long(), min=0))  
        return loss

class hinge_onevsrest_v1(nn.Module):
    def __init__(self,margin,interclass_loss=True):
        super(hinge_onevsrest_v1,self).__init__()
        self.margin = margin
        self.interclass_loss = interclass_loss
    def forward(self,outputs,labels):
        batch_size,num_classifier = outputs.shape
        mask = labels>=0
        concat_labels = torch.from_numpy(np.where(mask.cpu(),1,-1)).cuda()
        bin_loss = torch.clamp(self.margin - outputs.t()*concat_labels.t(), min=0)
        return bin_loss.mean()