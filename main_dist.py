
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import time
import json
import glob
import numpy as np
import torch
import torch.utils.data as D
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import transforms as T
from backbones.model_irse import IR_50
import modules.utils_torchvision as utils
from modules.utils_mine import accuracy_l2, get_l2_pred
from modules.utils_faceevolve import separate_irse_bn_paras, perform_val, get_val_pair,make_weights_for_balanced_classes
from torch.utils.tensorboard import SummaryWriter
from heads.epcc import DC_EPCC
from losses.hinge_loss import hinge_mc_v3
from losses.center_reg import center_regressor

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ["OMP_NUM_THREADS"]=str(3)

def train_one_epoch(model, head, criterion, criterion_center, optimizer, optimizer_center, data_loader, device, epoch, args):
    model.train(), head.train(), criterion.train(), criterion_center.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.1f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        
        if args.distributed:
            centers = criterion_center.module.centers
        else:
            centers = criterion_center.centers
       
        features = model(image)

        if(epoch>args.only_centers_upto):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_center.step()
        else:
            optimizer_center.zero_grad()
            loss.backward()
            optimizer_center.step()
        
        acc1 = accuracy_l2(features,centers,target[:,0])
        
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(),interclass_center_loss=interclass_center_loss.item(),intraclass_loss=intraclass_center_loss.item(),interclass_triplet=interclass_triplet.item(),lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
    
    metric_logger.synchronize_between_processes()
    print(' *Train Acc@1 {top1.global_avg:.3f} '
          .format(top1=metric_logger.acc1))
    return metric_logger.acc1.global_avg, metric_logger.interclass_center_loss.global_avg,metric_logger.intraclass_loss.global_avg,metric_logger.loss.global_avg

def load_data(args):
    # Data loading code
    print("Loading data")
    st = time.time()

    transform = T.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        T.Resize([int(128 * 112 / 112), int(128 * 112 / 112)]), # smaller side resized
        T.RandomCrop([112, 112]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(os.path.join(args.data_root, 'imgs'), transform)

    print("Took", time.time() - st)
   
    print("Creating data loaders")
    if args.distributed:
        train_sampler = D.distributed.DistributedSampler(dataset)
    else:
        weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
        weights = torch.DoubleTensor(weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return dataset, train_sampler

def main(args):
    save_dir = os.path.join('logs','casia-lfw','lr%.7fcentersupto%dintrak%.1finterk%.1ftripletk%.1frandomcenters%s_%s'%(args.lr,args.only_centers_upto,args.intrak,args.interk,args.tripletk,bool(args.train_centers_path),args.output_note))
    utils.mkdir(save_dir)
    with open(os.path.join(save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    utils.init_distributed_mode(args)
    
    print(args)
    
    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(12345)
    
    dataset, train_sampler = load_data(args)
    lfw, lfw_issame = get_val_pair('./data','lfw')

    data_loader = D.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,drop_last=True)

    print("Creating model")

    model = IR_50(pretrained=True,input_size=[112,112])
    #model = resnet50(pretrained=True,input_size=112,vis2d=False)
    model.to(device)

    head = DC_EPCC(model.embed_size,len(dataset.classes),kapa=args.kapa)
    head.to(device)

    criterion = hinge_mc_v3(args.margin)
    criterion.to(device)

    criterion_center = center_regressor(model.embed_size,len(dataset.classes),mutex_label=False)
    criterion_center.to(device)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(model) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability

    optimizer = torch.optim.Adam([{'params': backbone_paras_wo_bn+head.parse_params()['w'], 'weight_decay': args.weight_decay}, {'params': backbone_paras_only_bn+head.parse_params()['wabs']}], lr = args.lr)
    #optimizer = torch.optim.SGD([{'params': backbone_paras_wo_bn+head.parse_params()['w'], 'weight_decay': args.weight_decay}, {'params': backbone_paras_only_bn+head.parse_params()['wabs']}], lr = args.lr, momentum = args.momentum)

    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.5)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[int(args.epochs*0.6),int(args.epochs*0.8)])
    model_without_ddp = model
    head_without_ddp = head
    criterion_without_ddp = criterion
    criterion_center_without_ddp = criterion_center

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
        head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[args.gpu])
        head_without_ddp = head.module

        criterion = torch.nn.parallel.DistributedDataParallel(criterion, device_ids=[args.gpu])
        criterion_without_ddp = criterion.module

        criterion_center = torch.nn.parallel.DistributedDataParallel(criterion_center, device_ids=[args.gpu])
        criterion_center_without_ddp = criterion_center.module
    
    best_val_acc1_norm_lfw, best_val_acc1_lfw  = 0., 0.
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        head_without_ddp.load_state_dict(checkpoint['head'])
        criterion_without_ddp.load_state_dict(checkpoint['criterion'])
        criterion_center_without_ddp.load_state_dict(checkpoint['criterion_center'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_center.load_state_dict(checkpoint['optimizer_center'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_val_acc1_lfw = checkpoint['best_val_acc1_lfw']
        best_val_acc1_norm_lfw = checkpoint['best_val_acc1_norm_lfw']

    if args.test_only:
        print(perform_val(device, 512, 128, model, lfw, lfw_issame,normalized=True))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        acc1, loss = train_one_epoch(model, optimizer, criterion_center, optimizer_center, data_loader, device, epoch, dataset.num_subset, dataset.num_subcluster, args)
        
        if(epoch>args.only_centers_upto):
            lr_scheduler.step()
        
        if(epoch%args.eval_freq == 0 and epoch!=0):
            
            val_acc1_norm_lfw, _best_threshold_norm_lfw, _roc_curve_norm_lfw = perform_val(device, 512, 128, model, lfw, lfw_issame,normalized=True)
            is_best_norm_lfw = val_acc1_norm_lfw > best_val_acc1_norm_lfw
            best_val_acc1_norm_lfw = max(val_acc1_norm_lfw, best_val_acc1_norm_lfw)
            print('VAL_ACC1_NORMALIZED_LFW: %.5f, BEST_VAL_ACC1_NORMALIZED_LFW: %.5f '%(val_acc1_norm_lfw,best_val_acc1_norm_lfw))

            val_acc1_lfw, _best_threshold_lfw, _roc_curve_lfw = perform_val(device, 512, 128, model, lfw, lfw_issame,normalized=False)
            is_best_lfw = val_acc1_lfw > best_val_acc1_lfw
            best_val_acc1_lfw = max(val_acc1_lfw, best_val_acc1_lfw)
            print('VAL_ACC1_LFW: %.5f, BEST_VAL_ACC1_LFW: %.5f '%(val_acc1_lfw,best_val_acc1_lfw))

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'head': head_without_ddp.state_dict(),
                'criterion':criterion_without_ddp.state_dict(),
                'criterion_center': criterion_center_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_center': optimizer_center.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'val_acc1_lfw': val_acc1_lfw,
                'best_val_acc1_lfw': best_val_acc1_lfw,
                'val_acc1_norm_lfw': val_acc1_norm_lfw,
                'best_val_acc1_norm_lfw': best_val_acc1_norm_lfw,
                'args': args}

            if(epoch%250 == 0):
                utils.save_on_master(
                    checkpoint,
                    os.path.join(save_dir, 'model_{}.pth'.format(epoch)))
            if(is_best_lfw):
                utils.save_on_master(
                    checkpoint,
                    os.path.join(save_dir, 'best_checkpoint.pth'))
                is_best_lfw=False
            if(is_best_norm_lfw):
                utils.save_on_master(
                    checkpoint,
                    os.path.join(save_dir, 'best_checkpoint_norm.pth'))
                is_best_norm_lfw=False
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', default=751, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--only-centers-upto', default=-1, type=int, metavar='N',
                        help='train only centers up to epoch #')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=350, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--kapa', default=1.0, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-freq', default=30, type=int, help='print frequency')
    parser.add_argument('--folder-path', default='./data/esogu_faces', help='additional note to output folder')
    parser.add_argument('--output-note', default='subc5', help='additional note to output folder')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        default=False,
        help="Only test the model",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)