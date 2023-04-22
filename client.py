import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
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
        samples = 0.
        cumulative_loss = 0.
        for _, (images, labels) in enumerate(self.train_loader):
            outputs = self._get_outputs(images) # Apply the loss
            loss = self.criterion(outputs,labels) # Reset the optimizer
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            samples+=images.shape[0]
            cumulative_loss += loss.item()
        print(f"Loss at epoch {cur_epoch}: {cumulative_loss/samples}")

    def get_optimizer(net, lr, wd, momentum):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        return optimizer

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, weight_decay=0.00001, momentum=0.9)

        #optimizer = self.get_optimizer(self.model.parameters(), 0.01, 0.000001, 0.9)
        # TODO: missing code here!
        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer)
        raise NotImplementedError

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        # TODO: missing code here!
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                # TODO: missing code here!
                raise NotImplementedError
                self.update_metric(metric, outputs, labels)