import itertools, imageio, torch, random
import numpy as np
import torch.nn as nn
from torchvision import datasets
from scipy.misc import imresize
from torch.autograd import Variable
import yaml
import os

def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_() if m.bias is not None else None
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def read_yaml(config_weight):
    try:
        with open(config_weight) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except ValueError:
        print("No config file!")
    return config

def save_yaml(config, path):
    with open(path, 'w') as file:
        yaml.dump(config, file)

def print_config(config):
    print('------------ Options -------------')
    for k in config:
        print('{}: {}'.format(str(k), str(config[k])))
    print('-------------- End ----------------')

def get_device(config):
    if config['is_cuda'] == 'False':
        device = 'cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

    return device

def check_and_make_folfer(path):
    if not os.path.isdir(path):
        os.makedirs(path)
