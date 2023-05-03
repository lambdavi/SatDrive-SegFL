import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.utils import HardNegativeMining, MeanReduction
from torch.optim.lr_scheduler import StepLR, LinearLR 

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    @staticmethod
    def print_step_loss(losses, step):
        for name, l in losses.items():
            print(f"Train_{name}: {l} at step:{step}")

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        
        for cur_step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = self._get_outputs(images)
            loss = self.reduction(self.criterion(outputs,labels),labels)
            loss.backward()
            # Update parameters
            optimizer.step()
            

        print(f"\tLoss value at epoch {cur_epoch}/{self.args.num_epochs}: {loss.item()}")

    def get_optimizer_and_scheduler(self):
         # Optimizer chocie
        if self.args.opt == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.m)
        elif self.args.opt == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), eps=10**(-1), weight_decay=self.args.wd)
        else:
            raise NotImplementedError
        
        # Scheduler choice
        if self.args.sched == "lin":
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=10, verbose=True)
        elif self.args.sched == "step":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)
        else:
            scheduler = None

        return optimizer, scheduler

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        
        optimizer, scheduler = self.get_optimizer_and_scheduler()

        self.model.train()
        print("-----------------------------------------------------")
        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer)

            if scheduler:
                scheduler.step()
        print("-----------------------------------------------------")
        return len(self.dataset), self.model.state_dict()

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = self._get_outputs(images) # Apply the loss
                self.update_metric(metric, outputs, labels)
