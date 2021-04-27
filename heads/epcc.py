#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn import init
from modules.utils import epcc_kaiming_uniform_

__all__ = ['EPCC','EPCC_Ext','DC_EPCC','EPCC_Ext_Inverted']

class EPCC_Ext(nn.Module):   
    def __init__(self,embed_size,kapa=0.45):
        super(EPCC_Ext,self).__init__()
        self.embed_size = embed_size
        self.kapa = kapa
        self.w = nn.Linear(self.embed_size,1,bias=False)
        # wabs is indicated as gama in paper
        self.wabs = nn.Linear(self.embed_size,1,bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        init.uniform_(self.bias, 0.0, 1.0)
    def forward(self,x,center):
        xc = x - center
        y = self.w(xc) + self.wabs(xc.abs()) + self.bias.clamp(min=0.0)
        # y = self.w(xc) + self.wabs(xc.abs()) + 1.0
        return y
    
    def get_gama_reg_loss(self):
        gama_constraint = (self.kapa-(-self.wabs.weight-self.w.weight.abs())).clamp(min=0).mean()
        return gama_constraint
    
    def parse_params(self):
        w_param = list()
        wabs_param = list()
        for name, param in self.named_parameters():
            print(name)
            if('wabs' in name):
                wabs_param.append(param)
            else:
                w_param.append(param)
        return {'w':w_param,'wabs':wabs_param}
    
class DC_EPCC(nn.Module):
    def __init__(self,embed_size,num_classes,kapa=0.45):
        super(DC_EPCC,self).__init__()
        self.num_classes = num_classes
        self.kapa = kapa
        self.embed_size = embed_size
        self.epccs = nn.ModuleList([EPCC_Ext(embed_size,kapa) for i in range(self.num_classes)])
        self.register_buffer('centers', (
                torch.rand(num_classes, embed_size) - 0.5) * 2)
        
    def forward(self,x):
        self.outputs = [None] * self.num_classes
        for i, epcc in enumerate(self.epccs):
            self.outputs[i] = epcc(x,self.centers[i])
        self.outputs = torch.cat(self.outputs, dim=1)
        return self.outputs
    
    # def get_gama_reg_loss(self):
    #     gama_constraints = torch.zeros(self.num_classes).cuda()
    #     for i, epcc in enumerate(self.epccs):
    #         gama_constraints[i] = epcc.get_gama_reg_loss()
    #     return gama_constraints.mean()
    
    def get_gama_reg_loss(self):
        gama_constraints = torch.zeros(self.num_classes).cuda()
        for i in range(self.num_classes):
            gama_constraints[i] = (self.kapa-(-self.epccs[i].wabs.weight-self.epccs[i].w.weight.abs())).clamp(min=0).mean()
        return gama_constraints.mean()
    
    def parse_params(self):
        w_param = list()
        wabs_param = list()
        for name, param in self.named_parameters():
            print(name)
            if('wabs' in name):
                wabs_param.append(param)
            else:
                w_param.append(param)
        return {'w':w_param,'wabs':wabs_param}    

class EPCC(nn.Module):
    def __init__(self,embed_size,num_classes,kapa=0.45):
        super(EPCC,self).__init__()
        self.num_classes = num_classes
        self.kapa = kapa
        self.embed_size = embed_size
        self.epccs = nn.ModuleList([EPCC_Ext(embed_size,kapa) for i in range(self.num_classes)])
    
    def forward(self,x,centers):
        self.outputs = [None] * self.num_classes
        for i, epcc in enumerate(self.epccs):
            self.outputs[i] = epcc(x,centers)
        self.outputs = torch.cat(self.outputs, dim=1)
        return self.outputs
    
    def get_gama_reg_loss(self):
        gama_constraints = torch.zeros(self.num_classes).cuda()
        for i in range(self.num_classes):
            gama_constraints[i] = (self.kapa-(-self.epccs[i].wabs.weight-self.epccs[i].w.weight.abs())).clamp(min=0).mean()
        return gama_constraints.mean()
    
    def parse_params(self):
        w_param = list()
        wabs_param = list()
        for name, param in self.named_parameters():
            print(name)
            if('wabs' in name):
                wabs_param.append(param)
            else:
                w_param.append(param)
        return {'w':w_param,'wabs':wabs_param}   
    
    
class EPCC_Ext_Inverted(nn.Module):   
    def __init__(self,embed_size,num_classes=1,kapa=0.45):
        super(EPCC_Ext_Inverted,self).__init__()
        self.embed_size = embed_size
        self.kapa = kapa
        self.w = nn.Parameter(torch.Tensor(1, embed_size))
        self.wabs = nn.Parameter(torch.Tensor(1, embed_size))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.reset_parameters()
        
    def forward(self,x,center):
        xc = x - center
        # y = self.w(xc) + self.wabs(xc.abs()) - self.bias
        y = xc.matmul(self.w.t()) + xc.abs().matmul(self.wabs.t().clamp(min=0.0)) - self.bias.clamp(min=0.0)
        # y = xc.matmul(self.w.t()) + xc.abs().matmul(self.wabs.max(self.w.abs()+0.1).t()) - self.bias.clamp(min=0.0)
        return y
    
    def reset_parameters(self):
        epcc_kaiming_uniform_(self.w, a=math.sqrt(5))
        epcc_kaiming_uniform_(self.wabs, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, 0.0, bound)

    def get_gama_reg_loss(self):
        if(True):
            gama_constraints = (self.kapa-(self.wabs-self.w.abs())).clamp(min=0)
            gama_constraints = gama_constraints.mean()
        else:
            s_param = torch.Tensor().cuda().new_full((self.embed_size,1),self.kapa)
            # gama_constraints = torch.mean(self.wabs.weight.t()*s_param)
            gama_constraints = -torch.dot(self.wabs.squeeze(),s_param.squeeze())
        return gama_constraints

    def parse_params(self):
        w_param = list()
        wabs_param = list()
        for name, param in self.named_parameters():
            print(name)
            if('wabs' in name):
                wabs_param.append(param)
            else:
                w_param.append(param)
        return {'w':w_param,'wabs':wabs_param}

if __name__ == '__main__':
    epcc = EPCC_Ext_Inverted(3,1).cuda()
    epcc.parse_params()
    epcc(torch.ones(5,3).cuda(),torch.zeros(3).cuda())
    epcc.get_gama_reg_loss()
