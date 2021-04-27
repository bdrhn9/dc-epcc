#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import backbones
import heads
import losses
from modules.utils import AverageMeter, validate, save_checkpoint,accuracy,visualize,to_1vRest
from modules.dataset import dataset_loader
from options import get_options
from losses.center_reg import compute_center_loss,get_center_delta
args = get_options()

if(os.path.isdir(args.resume)):
    with open(os.path.join(args.resume,'commandline_args.txt'), 'r') as f:
        args.__dict__ = json.load(f)

# sync gpu order and bus order and select which to be used
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# create dataset and model
classes, num_classes, input_size, train_loader, val_loader = dataset_loader(args.batch_size,args.workers, args.dataset)

backbone = backbones.__dict__[args.backbone](pretrained=True,input_size=input_size,vis2d=args.vis2d)

if(args.onevsrest):
    head = to_1vRest(args.head,backbone.embed_size,num_classes)
elif(args.head in ['DC_EPCC','EPCC']):
    head = heads.__dict__[args.head](backbone.embed_size,num_classes,kapa=args.kapa)
elif(args.head in ['ArcMarginProduct','AddMarginProduct','SphereProduct','Linear_FC']):
    head = heads.__dict__[args.head](backbone.embed_size,num_classes)
else:
    raise('head is not defined')

backbone.cuda(), head.cuda()
cudnn.benchmark = True

# define optimizer
if(args.parse_epcc_params and args.head in ['DC_EPCC','EPCC']):
    params2optimize = [{'params':list(backbone.parameters())+head.parse_params()['w']},
                                    {'params':head.parse_params()['wabs'],'weight_decay':0.0}]
else:
    params2optimize = list(backbone.parameters())+list(head.parameters())

if(args.optimizer=='SGD'):
    optimizer_model = torch.optim.SGD(params2optimize, 
                                  args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
elif(args.optimizer=='Adam'):
    optimizer_model = torch.optim.Adam(params2optimize, 
                                  args.lr)
else:
    raise('optimizer is not defined')

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_model,
                                                    milestones=[int(args.epochs*0.6),int(args.epochs*0.8)])
# optionally resume from a checkpoint
#args.resume = "/home/bdrhn9/workspace_ml/dc-epcc/logs/SGD/cifar100/resnet50_DC_EPCC/hinge_mc_v2/exp3/"

if(os.path.isdir(args.resume)):
    checkpoint_path = os.path.join(args.resume,'best_checkpoint.pth')      
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    head.load_state_dict(checkpoint['head_state_dict'])
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    # centers_reg.load_state_dict(checkpoint['centers_reg_state_dict'])
    # optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
    # optimizer_centers_reg.load_state_dict(checkpoint['optimizer_centers_reg_state_dict'])
    best_val_top1 = checkpoint['best_val_top1']
    start_epoch = checkpoint['start_epoch']
    global_step = checkpoint['global_step']
else:
    print('start from scratch')
    best_val_top1, start_epoch, global_step = 0, 0, 0

if(args.loss in ['hinge','hinge_inverted',
                 'hinge_mc_v2','hinge_mc_v3','MultiMarginLoss',
                 'hinge_onevsrest_v0','hinge_onevsrest_v1']):
    criterion_model = losses.__dict__[args.loss](args.margin)
else:
    criterion_model = losses.__dict__[args.loss]()

# check the save_dir exists or not
save_dir = os.path.join('logs',args.optimizer,args.dataset,'%s_%s'%(args.backbone,args.head),'%s%s'%(args.loss,'_center_loss'if args.centerloss else ''))
exp_id = max([int(f.split('exp')[-1]) for f in glob.glob(save_dir + "/*")]+[0])+1
save_dir = os.path.join(save_dir,'exp%d'%(exp_id))
os.makedirs(save_dir,exist_ok=True)

#create summarywriter
writer = SummaryWriter(log_dir=save_dir)

# save args
with open(os.path.join(save_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

for epoch in range(start_epoch, args.epochs):
    # train for one epoch
    print('current lr {:.5e}'.format(optimizer_model.param_groups[0]['lr']))
    batch_time,model_losses = AverageMeter(), AverageMeter()
    gama_reg_losses,train_top1,center_reg_losses = AverageMeter(),AverageMeter(),AverageMeter()
    if(args.vis2d):
        all_features, all_labels = [], []
    # switch to train mode
    backbone.train(), head.train()
    end = time.time()
    epoch_time = time.time()
    for batch_id, (inputs, labels) in enumerate(train_loader):
       # measure data loading time
        inputs = inputs.cuda()
        labels = labels.cuda()

        # compute output
        features = backbone(inputs)
        # centers = centers_reg.centers
        
        if(args.onevsrest):
            outputs,model_loss = head(features,labels)
        else:
            if(args.head in ['DC_EPCC','EPCC']):
                outputs = head(features)
            elif(args.head in ['ArcMarginProduct','AddMarginProduct','SphereProduct']):
                outputs = head(features,labels)
            elif(args.head in ['Linear_FC']):
                outputs = head(features)
            else:
                raise('head is not defined')
            model_loss = criterion_model(outputs,labels)
        model_losses.update(model_loss.item(),inputs.size(0))
        
        if(args.head in ['DC_EPCC','EPCC']):
            gama_reg_loss = head.get_gama_reg_loss()
            gama_reg_losses.update(gama_reg_loss.item(),inputs.size(0))
            # epccs' gama regularization
            if(args.kapa != 0.0):
                model_loss = model_loss + gama_reg_loss

        # center regularization
        # center_reg_loss = centers_reg(features,labels)
        center_reg_loss = compute_center_loss(features,head.centers,labels)
        center_reg_losses.update(center_reg_loss.item(),inputs.size(0))
        
        # map feature embeddings like in center loss or nah
        if(args.centerloss):
            loss = model_loss + 0.003*center_reg_loss
            optimizer_centers_reg.zero_grad()
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            for param in centers_reg.parameters():
                param.grad.data *= (1. / 0.003)
            optimizer_centers_reg.step()
        else:
    
            optimizer_model.zero_grad()
            model_loss.backward()
            optimizer_model.step()
            center_deltas = get_center_delta(
                    features.data, head.centers, labels, 0.5,torch.device('cuda'))
            head.centers = head.centers - center_deltas
        
            
        # measure accuracy and record loss
        if(not args.onevsrest):
            """there is problem with one-vs-rest situation, it'll be solved in following commits"""
            train_prec1 = accuracy(outputs.data, labels)[0].item()
            train_top1.update(train_prec1, inputs.size(0))
        
        if(args.vis2d):
            all_features.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_id % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} '
                  'Model-Loss {model_loss.val:.4f} ({model_loss.avg:.4f}) '
                  'Center-Reg {center_reg.val:.4f} ({center_reg.avg:.4f}) '
                  'Gama-Reg {gama_reg.val:.4f} ({gama_reg.avg:.4f}) '
                  'Train-Prec@1 {train_top1.val:.3f} ({train_top1.avg:.3f})'.format(
                      epoch, batch_id, len(train_loader), batch_time=batch_time,
                      model_loss=model_losses,center_reg=center_reg_losses,
                      gama_reg=gama_reg_losses, train_top1=train_top1))
            writer.add_scalar('train/model_loss', model_losses.val,global_step)
            writer.add_scalar('train/accuracy', train_top1.val,global_step)
            writer.add_scalar('train/center_reg', center_reg_losses.val,global_step)
            writer.add_scalar('train/gama_reg', gama_reg_losses.val,global_step)
        global_step+=1
    print("Elapsed time for epoch %d, %.3f"%(epoch,time.time()-epoch_time))
    lr_scheduler.step()
    
    if(args.vis2d):
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        visualize(all_features, all_labels, num_classes,epoch, writer, args.head)
    
    # evaluate on validation set
    val_top1, val_classbased_ap, val_losses, val_features, val_labels, val_outputs = validate(val_loader, backbone, head,head.centers, criterion_model,args)
    writer.add_scalar('test/model_loss',val_losses.avg,global_step)
    writer.add_scalar('test/accuracy',val_top1.avg,global_step)
    # remember best prec@1 and save checkpoint
    is_best = val_top1.avg > best_val_top1
    best_val_top1 = max(val_top1.avg, best_val_top1)
    print('Epoch:%d train_top1:%.3f val_top1:%.3f best_val_top1:%.3f'%(epoch,train_top1.avg,val_top1.avg,best_val_top1))
    
    save_state = {  'head_state_dict': head.state_dict(),
                    'backbone_state_dict': backbone.state_dict(),
                    # 'centers_reg_state_dict': centers_reg.state_dict(),
                    'optimizer_model_state_dict': optimizer_model.state_dict(),
                    # 'optimizer_centers_reg_state_dict': optimizer_centers_reg.state_dict(),
                    'best_val_top1': best_val_top1,
                    'val_top1': val_top1.avg,
                    'val_classbased_ap':val_classbased_ap,
                    'train_top1': train_top1.avg,
                    'start_epoch': epoch + 1,
                    'global_step':global_step}
    
    # save_checkpoint(save_state,False,save_dir)
    if is_best:
        writer.add_scalar('test/best_accuracy',best_val_top1,global_step)
        save_checkpoint(save_state,is_best,save_dir)
writer.close()
