# -*- coding: utf-8 -*-

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader,Subset
from modules.voc import Voc2007Classification

def dataset_loader(batch_size, num_workers, dataset = 'cifar10',selected_cls_for_binary=2):
    print('current dataset: %s'%(dataset))
    if dataset == 'cifar10':
        input_size = 32
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])
        dataset_train = datasets.CIFAR10(root='./data', train=True, 
                                         transform=train_transform, download=True)
        dataset_test = datasets.CIFAR10(root='./data', 
                                        train=False,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                        normalize]))
        classes = dataset_train.classes
        num_classes = len(classes)
    
    elif dataset == 'cifar10_balanced_binary':
        input_size = 32
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])
        dataset_train = datasets.CIFAR10(root='./data', train=True, 
                                         transform=train_transform, download=True)
        dataset_test = datasets.CIFAR10(root='./data', 
                                        train=False,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                        normalize]))
        classes = dataset_train.classes
        num_classes = len(classes)
        dataset_train.targets = torch.Tensor(dataset_train.targets)
        dataset_test.targets = torch.Tensor(dataset_test.targets)
        indices_train = np.argwhere(dataset_train.targets==selected_cls_for_binary) # selected class
        n_samples_unselected_train = int(np.argwhere((dataset_train.targets==selected_cls_for_binary)).shape[-1]/(num_classes-1))
        for i in range(num_classes):
            if(i!=selected_cls_for_binary):
                indices_train=torch.cat((indices_train,np.argwhere(dataset_train.targets==i)[:,:n_samples_unselected_train]),1)
        
        indices_test = np.argwhere(dataset_test.targets==selected_cls_for_binary) # selected class
        n_samples_unselected_test = int(np.argwhere((dataset_test.targets==selected_cls_for_binary)).shape[-1]/(num_classes-1))
        for i in range(num_classes):
            if(i!=selected_cls_for_binary):
                indices_test=torch.cat((indices_test,np.argwhere(dataset_test.targets==i)[:,:n_samples_unselected_test]),1)
        
        dataset_train = Subset(dataset_train, indices_train.squeeze())
        dataset_test = Subset(dataset_test, indices_test.squeeze())  
        num_classes = 1

    elif dataset == 'cifar10_openset':
        input_size = 32
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])
        dataset_train = datasets.CIFAR10(root='./data', train=True, 
                                         transform=train_transform, download=True)
        dataset_test = datasets.CIFAR10(root='./data', 
                                        train=False,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                        normalize]))
        classes = dataset_train.classes[0:3]
        num_classes = len(classes)
        
        indices_train = np.argwhere((np.array(dataset_train.targets)==0) | (np.array(dataset_train.targets)==1) |(np.array(dataset_train.targets)==2))[:,0]
        dataset_train = Subset(dataset_train, indices_train)

    elif dataset =='cifar100':
        input_size = 32
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])
        dataset_train = datasets.CIFAR100(root='./data', train=True, 
                                            transform=train_transform, download=True)
        dataset_test = datasets.CIFAR100(root='./data', 
                                         train=False,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                        normalize]))
        classes = dataset_train.classes
        num_classes = len(classes)
        
    elif dataset == 'mnist':
        input_size = 28
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root='./data', train=True, 
                                       transform=transform, download=True)
        dataset_test = datasets.MNIST(root='./data', train=False,
                                      transform=transform, download=True)
        
        classes = dataset_train.classes
        num_classes = len(classes)
    
    elif dataset == 'mnist_two_cls':
        'number 0,1 are only used'
        input_size = 28
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root='./data', train=True, 
                                       transform=transform, download=True)
        dataset_test = datasets.MNIST(root='./data', train=False,
                                      transform=transform, download=True)
        
        classes = dataset_train.classes[0:2]
        num_classes = len(classes)
        
        indices_train = np.argwhere((dataset_train.targets==0) | (dataset_train.targets==1))[0]
        dataset_train = Subset(dataset_train, indices_train)
        
        indices_test = np.argwhere((dataset_test.targets==0) | (dataset_test.targets==1))[0]
        dataset_test = Subset(dataset_test, indices_test)
        
    elif dataset == 'mnist_three_cls':
        'number 0,1,2 are only used'
        input_size = 28
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root='./data', train=True, 
                                       transform=transform, download=True)
        dataset_test = datasets.MNIST(root='./data', train=False,
                                      transform=transform, download=True)
        classes = dataset_train.classes[0:3]
        num_classes = len(classes)
        
        indices_train = np.argwhere((dataset_train.targets==0) | (dataset_train.targets==1) |(dataset_train.targets==2))[0]
        dataset_train = Subset(dataset_train, indices_train)
        
        indices_test = np.argwhere((dataset_test.targets==0) | (dataset_test.targets==1) |(dataset_test.targets==2))[0]
        dataset_test = Subset(dataset_test, indices_test)
        
    elif dataset == 'mnist_five_cls':
        'number 0,1,2,3,4 are only used'
        input_size = 28
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root='./data', train=True, 
                                       transform=transform, download=True)
        dataset_test = datasets.MNIST(root='./data', train=False,
                                      transform=transform, download=True)
        classes = dataset_train.classes[0:5]
        num_classes = len(classes)
        
        indices_train = np.argwhere((dataset_train.targets==0) | (dataset_train.targets==1) | (dataset_train.targets==2) | (dataset_train.targets==3) | (dataset_train.targets==4))[0]
        dataset_train = Subset(dataset_train, indices_train)
        
        indices_test = np.argwhere((dataset_test.targets==0) | (dataset_test.targets==1) |(dataset_test.targets==2) | (dataset_test.targets==3) | (dataset_test.targets==4))[0]
        dataset_test = Subset(dataset_test, indices_test)    
        
    elif dataset == 'mnist_balanced_binary':
        input_size = 28
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root='./data', train=True, 
                                       transform=transform, download=True)
        dataset_test = datasets.MNIST(root='./data', train=False,
                                      transform=transform, download=True)
        classes = dataset_train.classes
        num_classes = len(classes)
        
        indices_train = np.argwhere(dataset_train.targets==selected_cls_for_binary) # selected class
        n_samples_unselected_train = int(np.argwhere((dataset_train.targets==selected_cls_for_binary)).shape[-1]/(num_classes-1))
        for i in range(num_classes):
            if(i!=selected_cls_for_binary):
                indices_train=torch.cat((indices_train,np.argwhere(dataset_train.targets==i)[:,:n_samples_unselected_train]),1)
        
        indices_test = np.argwhere(dataset_test.targets==selected_cls_for_binary) # selected class
        n_samples_unselected_test = int(np.argwhere((dataset_test.targets==selected_cls_for_binary)).shape[-1]/(num_classes-1))
        for i in range(num_classes):
            if(i!=selected_cls_for_binary):
                indices_test=torch.cat((indices_test,np.argwhere(dataset_test.targets==i)[:,:n_samples_unselected_test]),1)
        
        dataset_train = Subset(dataset_train, indices_train.squeeze())
        dataset_test = Subset(dataset_test, indices_test.squeeze())  
        num_classes = 1
    
    elif dataset == 'mnist_binary':
        'number 3 is selected as positive, another ones are negative'
        input_size = 28
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root='./data', train=True, 
                                       transform=transform, download=True)
        dataset_test = datasets.MNIST(root='./data', train=False,
                                      transform=transform, download=True)
        dataset_train.targets[dataset_train.targets!=selected_cls_for_binary] = -1
        dataset_test.targets[dataset_test.targets!=selected_cls_for_binary] = -1
        dataset_train.targets[dataset_train.targets==selected_cls_for_binary] = 1
        dataset_test.targets[dataset_test.targets==selected_cls_for_binary] = 1
        
        classes = dataset_train.classes[selected_cls_for_binary]
        num_classes = 1 
    
    elif dataset =='facescrub':
        input_size = 112
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        train_transform = transforms.Compose([transforms.Resize((112,112)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        dataset_train = datasets.ImageFolder('./data/facescrub/train',train_transform)
        dataset_test = datasets.ImageFolder('./data/facescrub/test',
            transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor(),normalize]))
        classes = dataset_train.classes
        num_classes = len(classes)  
    
    elif dataset =='voc2007':
        input_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
        dataset_train = Voc2007Classification(root='./data/VOCtrainval_06-Nov-2007', set='trainval',
                                         transform=train_transform)
        dataset_test = Voc2007Classification(root='./data/VOCtest_06-Nov-2007', set='test',
                                        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                                                        normalize]))
        classes = dataset_train.classes
        num_classes = len(classes)
        
    else:
        raise('unknown_dataset')
    
    train_loader = DataLoader(dataset_train,
    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(dataset_test,
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return classes, num_classes, input_size, train_loader, test_loader

if __name__ =='__main__':
    mnist_loader = dataset_loader(64, 4,'facescrub')
