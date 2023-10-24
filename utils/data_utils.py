import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from skimage.transform import resize
from timm.data import Mixup
from timm.data import create_transform


import torch
from torchvision import transforms

import torch.utils.data as data

Image.LOAD_TRUNCATED_IMAGES = True

CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


class DatasetFLViT(data.Dataset):
    def __init__(self, args, loaded_npy, phase):
        super(DatasetFLViT, self).__init__()
        self.phase = phase
        self.loaded_npy = loaded_npy

        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:

            self.transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])


        data_all = self.loaded_npy
        data_all = data_all.item()
        self.data_all = data_all[args.split_type]

        if self.phase == 'train':
            if args.dataset == 'cifar10' or args.dataset == "pacs" :
                self.data = self.data_all['data'][args.single_client]
                self.labels = self.data_all['target'][args.single_client]

            elif args.split_type == 'central'  and args.dataset in ['celeba', 'gldk23', 'isic19']: 
                self.data =  list(self.data_all['train'].keys())
                self.labels = data_all['labels']

            else:
                self.data = self.data_all['train'][args.single_client]['x']
                self.labels = data_all['labels']
        
        else:
            if args.dataset == 'cifar10' or args.dataset == "pacs":
                self.data = data_all['union_' + phase]['data']
                self.labels = data_all['union_' + phase]['target']

            elif args.split_type == 'central' and args.dataset in ['celeba', 'gldk23', 'isic19']: 

                if phase == 'val':
                    val_all = []
                    for client in data_all['real']['val'].keys():
                        val_all.extend(data_all['real']['val'][client]['x'])

                    self.data = val_all
                    self.labels = data_all['labels']
                
                elif phase == 'test': 
                    self.data = list(data_all['central']['val'].keys())
                    self.labels = data_all['labels']

            else:
                if args.split_type == 'real' and phase == 'val':
                    self.data = self.data_all['val'][args.single_client]['x']

                elif args.split_type == 'central' or phase == 'test':
                    self.data = list(data_all['central']['val'].keys())

                self.labels = data_all['labels']

        self.args = args


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.args.dataset == 'cifar10' or self.args.dataset == "pacs":
            img, target = self.data[index], self.labels[index]
            img = Image.fromarray(img)

        elif self.args.dataset == 'celeba':
            name = self.data[index]
            target = self.labels[name]
            path = os.path.join(self.args.data_path, self.args.dataset, 'celeba_images', name)
            img = Image.open(path).convert('RGB')
            target = np.asarray(target).astype('int64')

        elif self.args.dataset  == 'gldk23':
            name = self.data[index]
            target = self.labels[name]
            path = os.path.join(self.args.data_path, self.args.dataset, 'gldk23_images', name)
            img = Image.open(path).convert('RGB')
            target = np.asarray(target).astype('int64')

        elif self.args.dataset == 'isic19': 
            name = self.data[index]
            target = self.labels[name]
            path = os.path.join(self.args.data_path, self.args.dataset, 'isic19_images', name)
            img = Image.open(path).convert('RGB')
            target = np.asarray(target).astype('int64')           

        if self.transform is not None:
            img = self.transform(img)

        return img,  target


    def __len__(self):
        return len(self.data)


def create_dataset_and_evalmetrix(args):

    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = ['central']

    if args.dataset == 'cifar10' or args.dataset == "pacs" :

        # get the client with number
        print('Loading dataset and npy file')
        data_all_loaded = np.load(os.path.join(args.data_path, args.dataset, args.dataset + '.npy'), allow_pickle=True)
        data_all = data_all_loaded.item()

        data_all = data_all[args.split_type]
        args.dis_cvs_files = [key for key in data_all['data'].keys() if 'train' in key]
        args.clients_with_len = {name: data_all['data'][name].shape[0] for name in args.dis_cvs_files}


    elif args.dataset in ['celeba', 'gldk23', 'isic19']:
        data_all_loaded = np.load(os.path.join(args.data_path, args.dataset, args.dataset + '.npy'), allow_pickle=True)
        data_all = data_all_loaded.item()
        args.dis_cvs_files = list(data_all[args.split_type]['train'].keys())

        if args.split_type == 'real':
            args.clients_with_len = {name: len(data_all['real']['train'][name]['x']) for name in
                                     data_all['real']['train']}
            
        elif  args.split_type == 'central':
            args.dis_cvs_files = ['central']
            args.clients_with_len = {'central' : len(data_all['central']['train'])}


    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc_avg = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}

    for single_client in args.dis_cvs_files:
        args.best_acc[single_client] = 0 if args.num_classes > 1 else 999
        args.current_acc[single_client] = []
        args.current_test_acc[single_client] = []
        args.best_eval_loss[single_client] = 9999

    return data_all_loaded