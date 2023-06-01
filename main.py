import os
import json
from collections import defaultdict

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr
import datasets.weather as weather

from torch import nn
from client import Client
from server import Server
from fda_server import FdaServer
from utils.args import get_parser
from utils.utils import split_list_random, split_list_balanced
from datasets.idda import IDDADataset
from datasets.loveda import LoveDADataset
from datasets.gta5 import GTA5Dataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from models.bisenetv2 import BiSeNetV2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from transformers import SegformerForSemanticSegmentation

from torchvision.transforms import RandomApply

import timeit
import os

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
    if dataset == 'gta5':
        return 20
    if dataset == 'femnist':
        return 62
    if dataset == 'loveda':
        return 7
    raise NotImplementedError

def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'segformer':
        weights = args.transformer_model
        return SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/mit-{weights}",
            num_labels=get_dataset_num_classes(args.dataset),
            ignore_mismatched_sizes=True,
        )
    if args.model == "bisenetv2":
            return BiSeNetV2(get_dataset_num_classes(args.dataset), pretrained=True)

    raise NotImplementedError

def get_transforms(args):
    if args.model in ["segformer",'deeplabv3_mobilenetv2']:
        train_transforms = [
            sstr.Compose([
                RandomApply([sstr.Lambda(lambda x: weather.add_rain(x))], p=0.15),
            ]),
            sstr.Compose([
                sstr.RandomCrop((512, 928)),
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]), 
        ]
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    elif args.model == "bisenetv2":
        train_transforms = sstr.Compose([
                sstr.RandomCrop((512, 928)),
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms

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

    elif args.dataset == 'gta5':
        root = 'data/gta5'

        # Extract all data from train.txt
        all_data_train = []
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            all_data_train = f.read().splitlines()
        f.close()

        print(f"Total number of images to be loaded: {len(all_data_train)}")
        
        if args.centr:
            # If centralized we get all training data on one single client
            print("Centralized mode set.")
            train_datasets.append(GTA5Dataset(root=root, list_samples=all_data_train, transform=train_transforms,
                                                client_name='centralized'))
        else:
            # Otherwise we divide data in multiple datasets.
            print("Distributed Mode Set.")

            total_client_splits = split_list_balanced(all_data_train, args.clients_per_round*2)
            
            for i, samples in enumerate(total_client_splits):
                train_datasets.append(GTA5Dataset(root=root, list_samples=samples, transform=train_transforms,
                                                client_name="client_"+str(i)))
        root_idda = "data/idda"

        # Test on IDDA
        with open(os.path.join(root_idda, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root_idda, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root_idda, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root_idda, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]
        
        # Setting up IDDA as validation set
        validation_data = []
        with open(os.path.join(root_idda, 'train.txt'), 'r') as f:
            all_data = f.read().splitlines()
        validation_data.append(IDDADataset(root=root_idda, list_samples=all_data, transform=train_transforms,
                                             client_name='centralized'))
        
        
        return train_datasets, test_datasets, validation_data

    elif args.dataset == "loveda":
        root = 'data/loveda'

        # Extract all data from the Urban (trainset) 
        all_data_train = os.listdir(os.path.join(root, "Urban", "images_png"))

        print(f"Total number of images to be loaded: {len(all_data_train)}")
        
        if args.centr:
            # If centralized we get all training data on one single client
            print("Centralized mode set.")
            train_datasets.append(LoveDADataset(root=root, list_samples=all_data_train, folder="Urban", transform=train_transforms,
                                                client_name='centralized'))
        else:
            # Otherwise we divide data in multiple datasets.
            print("Distributed Mode Set.")

            total_client_splits = split_list_balanced(all_data_train, args.clients_per_round*2)
            
            for i, samples in enumerate(total_client_splits):
                train_datasets.append(LoveDADataset(root=root, list_samples=samples, folder="Urban", transform=train_transforms,
                                                client_name="client_"+str(i)))

        # Test on Rural
        test_same_dom_data = os.listdir(os.path.join(root, "Urban2", "images_png"))
        test_same_dom_dataset = LoveDADataset(root=root, list_samples=test_same_dom_data, folder="Urban2", transform=test_transforms,
                                                client_name='test_same_dom')
        
        test_diff_dom_data = os.listdir(os.path.join(root, "Rural", "images_png"))
        test_diff_dom_dataset = LoveDADataset(root=root, list_samples=test_diff_dom_data, folder="Rural", transform=test_transforms,
                                            client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]
        
    else:
        raise NotImplementedError

    return train_datasets, test_datasets, None

def get_source_client(args, model, same=None):
    train_transforms, _ = get_transforms(args)
    if args.fda:
        if args.dataset == "idda": # target == idda
            root = 'data/gta5'
            # Extract all data from train.txt
            all_data_train = []
            with open(os.path.join(root, 'train.txt'), 'r') as f:
                all_data_train = f.read().splitlines()
            f.close()
            sc = Client(args, GTA5Dataset(root=root, list_samples=all_data_train, transform=train_transforms, client_name='gta5_all'), model)
            return [sc]
        elif args.dataset == "loveda":
            return [same]
    else:
        return None

def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model in ['deeplabv3_mobilenetv2', "segformer", "bisenetv2"]:
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

def gen_clients(args, train_datasets, test_datasets, validation_datasets, model):
    clients = [[], [], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        # For each dataset datasets (one for each client), create and append a client
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    if validation_datasets:
        clients[2].append(Client(args, validation_datasets[0], model, test_client=True))
    return clients[0], clients[1], clients[2]

def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    print('Initializing model...', end=" ")
    model = model_init(args)
    model.cuda()
    print('Done.')

    train_datasets, test_datasets, validation_dataset = get_datasets(args)
    print('Generate datasets...', end=" ")
    print('Done.')
    source_dataset = get_source_client(args, model, train_datasets)
    metrics = set_metrics(args)

    print('Generate clients...', end=" ")
    train_clients, test_clients, valid_clients = gen_clients(args, train_datasets, test_datasets, validation_dataset, model)
    print('Done.')

    print('Setup server...', end=" ")
    if args.fda == False:
        if args.dataset == "gta5":
            server = Server(args, train_clients, test_clients, model, metrics, True, valid_clients)
        else: 
            server = Server(args, train_clients, test_clients, model, metrics)
    else:
        print("\nActivating FDA mode...\t", end="")
        server = FdaServer(args, source_dataset, train_clients, test_clients, model, metrics)
    print('Done.')

    execution_time = timeit.timeit(server.train, number=1)
    print(f"Execution time: {execution_time} seconds")

    # Code to predict an image
    if args.pred:
        print("Predicting "+args.pred)
        server.predict(args.pred)

if __name__ == '__main__':
    main()
