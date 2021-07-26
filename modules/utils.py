#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import time
import math
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from sklearn.metrics import average_precision_score
from modules.vis_decision_regions import plot_decision_regions_epcc,plot_decision_regions_feats
import heads
import losses

def accuracy_l2(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)

        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size) 

def get_l2_pred(features,centers):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)

        return pred

def epcc_kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only 
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(0.0, bound)

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def save_checkpoint(state, is_best, save_dir):
    """
    Save the training model
    """
    print('Checkpoint saved w/ Val.Acc.:%.3f Best.Val.Acc.:%.3f'%(state['val_top1'],state['best_val_top1']))
    torch.save(state, os.path.join(save_dir, 'best_checkpoint.pth' if is_best else 'checkpoint.pth'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, backbone, head,centers, criterion_model , args):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    backbone.eval(), head.eval()
    all_features, all_labels, all_outputs = [], [], []
    end = time.time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            all_labels.append(labels.data.cpu().numpy())

            # compute output
            features = backbone(inputs)
            if(args.onevsrest):
                if(args.centerloss):
                    outputs,loss,closs = head(features,labels)
                else:
                    outputs,loss = head(features,labels)
            else:
                if(args.head in ['DC_EPCC','EPCC']):
                    outputs = head(features,centers)
                elif(args.head in ['ArcMarginProduct','AddMarginProduct','SphereProduct']):
                    outputs = head(features,labels)
                elif(args.head in ['Linear_FC']):
                    outputs = head(features)
                else:
                    raise('head is not defined')
                loss = criterion_model(outputs,labels)
            losses.update(loss.item(), inputs.size(0))

            all_features.append(features.data.cpu().numpy())
            all_outputs.append(outputs.data.cpu().numpy())
            if(not args.onevsrest):
                # measure multi_class accuracy
                prec1 = accuracy(outputs.data, labels)[0].item()
                top1.update(prec1, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
    val_features = np.concatenate(all_features, 0)
    val_labels = np.concatenate(all_labels, 0)
    val_outputs = np.concatenate(all_outputs, 0)
    classbased_ap = None
    if(args.onevsrest):
        # measure accuracy onevsrest
        val_labels = np.where(val_labels>=0,1,0)
        prec1 = average_precision_score(val_labels,val_outputs)
        top1.avg = prec1
        classbased_ap = average_precision_score(val_labels,val_outputs,average=None)
    return top1, classbased_ap, losses, val_features, val_labels, val_outputs

def validate_one_class(val_loader, backbone,centers_reg, args):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    backbone.eval()
    centers_reg.eval()
    centers = centers_reg.centers
    all_features, all_labels, all_outputs = [], [], []
    end = time.time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            all_labels.append(labels.data.cpu().numpy())

            # compute output
            features = backbone(inputs)
            loss = centers_reg(features,labels)
            losses.update(loss.item(), inputs.size(0))

            all_features.append(features.data.cpu().numpy())
            all_outputs.append(outputs.data.cpu().numpy())
            prec1 = accuracy_l2(features,centers,labels)
            top1.update(prec1, inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
   
    return top1, losses

def plot_dec_boundary_2d(feats,labels,head,centers,save_path,binary,only_feats=False):
    plt.clf()
    if(only_feats):
        ax = plot_decision_regions_feats(feats, labels,legend=2,hide_spines=False)
    else:
        ax = plot_decision_regions_epcc(feats, labels, head=head,centers=centers,binary=binary,legend=2,hide_spines=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels, framealpha=0.3, scatterpoints=1)
    if(save_path):
        plt.savefig(save_path)

def visualize(feat, labels, num_classes, epoch,writer,name):
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
              '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    for i in range(num_classes):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.text(0, 0, "epoch=%d" % epoch)
    canvas.draw()

    # if (os.path.exists(imgDir)):
    #     pass
    # else:
    #     os.makedirs(imgDir)
    # fig.savefig(imgDir + '/epoch=%d.jpg' % epoch)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tt = transforms.ToTensor()
    timg = tt(img)
    timg.unsqueeze(0)
    writer.add_image(name, timg, epoch)
   
class to_1vRest(nn.Module):
    def __init__(self,head_name,embed_size,num_classes):
        super(to_1vRest,self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.exc_heads = nn.ModuleList([heads.__dict__[head_name](embed_size,2) for i in range(self.num_classes)])
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,x,labels):
        stacked_outputs = [None] * self.num_classes
        losses = torch.zeros(self.num_classes).cuda()
        labels = torch.from_numpy(np.where(labels.cpu().numpy()>=0,1.0,0.0)).cuda().detach()
        for i, exc_head in enumerate(self.exc_heads):
            stacked_outputs[i] = exc_head(x,labels[:,i].long())
            losses[i] = self.criterion(stacked_outputs[i],labels[:,i].long())
        stacked_outputs = torch.stack(stacked_outputs, dim=1)
        outputs = stacked_outputs[:,:,1]-stacked_outputs[:,:,0]
        return outputs,losses.mean()

class to_1vRest_center(nn.Module):
    def __init__(self,head_name,embed_size,num_classes):
        super(to_1vRest_center,self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.exc_heads = nn.ModuleList([heads.__dict__[head_name](embed_size,2) for i in range(self.num_classes)])
        self.cregs = nn.ModuleList([losses.__dict__['center_regressor'](embed_size,2,mutex_label=False) for i in range(self.num_classes)])
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,x,labels):
        stacked_outputs = [None] * self.num_classes
        losses = torch.zeros(self.num_classes).cuda()
        creg_losses = torch.zeros(self.num_classes).cuda()
        labels = torch.from_numpy(np.where(labels.cpu().numpy()>=0,1.0,0.0)).cuda().detach()
        for i,creg in enumerate(self.cregs):
            creg_losses[i] = creg(x,labels[:,i].long())
        
        for i, exc_head in enumerate(self.exc_heads):
            stacked_outputs[i] = exc_head(x,labels[:,i])
            losses[i] = self.criterion(stacked_outputs[i],labels[:,i].long())
        stacked_outputs = torch.stack(stacked_outputs, dim=1)
        outputs = stacked_outputs[:,:,1]-stacked_outputs[:,:,0]
        return outputs,losses.mean(),creg_losses.mean()
