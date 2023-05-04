import os
import json
from collections import defaultdict

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from client import Client
from datasets.femnist import Femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics

from torchvision.transforms import RandomApply
import cv2
from PIL import Image

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError

def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        # TODO: missing code here!
        raise NotImplementedError
    raise NotImplementedError


def get_transforms(args):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = [
            sstr.Compose([
                # TODO move weather conditions into a separate file
                RandomApply([sstr.Lambda(lambda x: add_rain(x))], p=0.3)
            ]),
            sstr.Compose([
                sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
                sstr.RandomHorizontalFlip(),
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]), 
        ]
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


# def read_femnist_dir(data_dir):
#     data = defaultdict(lambda: {})
#     files = os.listdir(data_dir)
#     files = [f for f in files if f.endswith('.json')]
#     for f in files:
#         file_path = os.path.join(data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         data.update(cdata['user_data'])
#     return data


# def read_femnist_data(train_data_dir, test_data_dir):
#     return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)

def generate_random_lines(imshape, slant, drop_length):
    drops = []
    num_drops = 1500
    x = np.random.randint(0, imshape[1], size=(num_drops,))
    y = np.random.randint(0, imshape[0]-drop_length, size=(num_drops,))
    if slant < 0:
        x += slant
    else:
        x -= slant
    drops = np.column_stack((x, y)).tolist()
    return drops

def add_rain(pil_image):
    # Convert PIL image to OpenCV image
    open_cv_image = np.array(pil_image)
    image = open_cv_image[:, :, ::-1].copy()
    
    # Generate random lines using numpy functions
    imshape = image.shape
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 20
    drop_width = 2
    drop_color = (200, 200, 200)
    x = np.arange(0, imshape[1], 10)
    y = np.random.randint(0, imshape[0], size=len(x))
    rain_drops = np.column_stack((x, y))
    
    # Draw rain drops using OpenCV function
    for rain_drop in rain_drops:
        cv2.line(image, (rain_drop[0], rain_drop[1]), 
                 (rain_drop[0]+slant, rain_drop[1]+drop_length), 
                 drop_color, drop_width)
    
    # Blur the image using OpenCV function
    image = cv2.blur(image, (7, 7))
    
    # Adjust the brightness of the image
    brightness_coefficient = 0.7
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image[:, :, 1] = image[:, :, 1] * brightness_coefficient
    image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    
    # Convert OpenCV image to PIL image and return
    im_pil = Image.fromarray(image)
    return im_pil
def get_datasets(args):

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'

        if args.centr:
          # If centralized we get all training data on one single client
          print("Centralized mode set")
          with open(os.path.join(root, 'train.txt'), 'r') as f:
            all_data = f.read().splitlines()
          train_datasets.append(IDDADataset(root=root, list_samples=all_data, transform=train_transforms,
                                             client_name='centralized'))
        else:
          # Otherwise we divide data in multiple datasets.
          print("Distributed Mode Set")

          with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
          for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    # elif args.dataset == 'femnist':
    #     niid = args.niid
    #     train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
    #     test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')
    #     train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)

    #     train_transforms, test_transforms = get_transforms(args)

    #     train_datasets, test_datasets = [], []

    #     for user, data in train_data.items():
    #         train_datasets.append(Femnist(data, train_transforms, user))
    #     for user, data in test_data.items():
    #         test_datasets.append(Femnist(data, test_transforms, user))

    else:
        raise NotImplementedError

    return train_datasets, test_datasets

def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        # For each dataset datasets (one for each client), create and append a client
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    return clients[0], clients[1]


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    print(f'Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    print('Generate datasets...')
    train_datasets, test_datasets = get_datasets(args)
    print('Done.')

    metrics = set_metrics(args)
    
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    server = Server(args, train_clients, test_clients, model, metrics)
    server.train()

if __name__ == '__main__':
    main()
