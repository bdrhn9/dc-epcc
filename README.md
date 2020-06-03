# Deep Polyhedral Conic Classifier for Open and Closed Set Recognition
## Introduction
In this paper, we propose a new deep neural network classifier that simultaneously 2 maximizes the inter-class separation and minimizes the intra-class variation by 3 using the polyhedral conic classification function. The proposed method has one 4 loss term that allows the margin maximization to maximize the inter-class separa5 tion and another loss term that controls the compactness of the class acceptance 6 regions. Our proposed method has a nice geometric interpretation using polyhedral 7 conic function geometry. We tested the proposed methods on various classification 8 problems. The experimental results show that the proposed method outperforms 9 other methods, and becomes a better choice compared to other tested methods 10 especially for open set recognition type problems.

## Train

First of all, create conda environment using environment.yml on CUDA-Enabled system..
> conda env create -f environment.yml

### Training on CIFAR-10, CIFAR-100, MNIST, FaceScrub datasets
(1.) Train DC_EPCC with ResNet18 on CIFAR-10
> python main.py --backbone resnet18 --head DC_EPCC --loss hinge_mc_v2 --kapa 1.0 --margin 1.0 --dataset cifar10

(2.) Train ArcFace with ResNet50 on CIFAR-100
> python main.py --backbone resnet50 --head ArcMarginProduct --loss CrossEntropyLoss --dataset cifar100

(3.) Train CosFace with IR-50 on FaceScrub
> python main.py --backbone IR_50 --head AddMarginProduct --loss CrossEntropyLoss --dataset facescrub

(4.) Train SphereFace with LeNet++ on MNIST
> python main.py --backbone LeNet --head SphereProduct --loss CrossEntropyLoss --dataset mnist

(5.) Train Softmax with ResNet50 on CIFAR-100
> python main.py --backbone resnet50 --head Linear_FC --loss CrossEntropyLoss --dataset cifar100

(6.) Train CenterLoss with ResNet101 on CIFAR-10
> python main.py --backbone resnet101 --head Linear_FC --loss CrossEntropyLoss --dataset cifar10 --centerloss

### Training on VOC2007 dataset
(1.) Train DC_EPCC with ResNet50
> python main.py --backbone resnet50 --head DC_EPCC --loss hinge_onevsrest_v1 --kapa 1.0 --margin 1.0 --dataset voc2007

(2.) Train ArcFace with ResNet50
> python main.py --backbone resnet50 --head ArcMarginProduct --dataset voc2007 --onevsrest

(3.) Train CosFace with ResNet50
> python main.py --backbone resnet50 --head AddMarginProduct --dataset voc2007 --onevsrest

(4.) Train SphereFace with ResNet50
> python main.py --backbone resnet50 --head SphereProduct --dataset voc2007 --onevsrest

(5.) Train Softmax with ResNet50
> python main.py --backbone resnet50 --head Linear_FC --dataset voc2007 --onevsrest

(6.) Train CenterLoss with ResNet50
> It will be updated in next commits.

## Pretrained Models
For the anonymity, gdrive links will be published in GitHub repo.
