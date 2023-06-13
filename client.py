import copy

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.early_stopping import EarlyStopper
from utils.loss import IW_MaxSquareloss, SelfTrainingLoss
from utils.utils import HardNegativeMining, MeanReduction, get_save_string

class Client:
    """
    Client class for training and testing models on a specific dataset.
    Args:
            `args`: Arguments object containing the client-specific configurations.\n
            `dataset`: Dataset object for training and testing.\n
            `model`: Model object to be trained and tested.\n
            `test_client` (bool): Flag indicating if the client is a test client.\n
            `val` (bool): Flag indicating if validation should be performed.
    """
    def __init__(self, args, dataset, model, test_client=False, val=False):
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

        self.styleaug = None
        self.early_stopper = EarlyStopper(args)
        
        self.teacher = None

        self.mious = [[], [], []]

    def __str__(self):
        """
        Return the name of the client as a string.
        """
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        """
        Update the evaluation metric with the model outputs and labels.

        Args:
            `metric`: Metric object to be updated.\n
            `outputs`: Model outputs.\n
            `labels`: True labels.

        """
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def set_teacher(self, teacher_model):
        """
        Set the teacher model for self-training.

        Args:
            `teacher_model`: Teacher model object.

        """
        self.teacher = copy.deepcopy(teacher_model)

    def _get_outputs(self, images, labels=None, test=False):
        """
        Get the model outputs for the given images and labels.

        Args:
            `images`: Input images.\n
            `labels`: True labels.\n
            `test` (bool): Flag indicating if the model is being tested.
        Returns:
            Model outputs.
        """
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model in ['resnet18',]:
            return self.model(images)
        if self.args.model == 'segformer':
            logits = self.model(images).logits
            outputs = nn.functional.interpolate(
                    logits, 
                    size=labels.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
            )
            return outputs
        if self.args.model == 'bisenetv2':
            outputs = self.model(images, test=test)
            return outputs
            
        raise NotImplementedError
    
    def __get_criterion_and_reduction_rules(self, use_labels=False):
        """
        Get the criterion and reduction rules for training.

        Args:
            `use_labels` (bool): Flag indicating if labels are used.

        Returns:
            Criterion object and reduction object.
        """
        shared_kwargs = {'ignore_index': 255, 'reduction': 'none'}
        if self.args.loss == "self":
            criterion = SelfTrainingLoss(lambda_selftrain=1, conf_th=self.args.pseudo_conf, fraction=self.args.frac,  **shared_kwargs)
            criterion.set_teacher(copy.deepcopy(self.teacher))
        elif self.args.loss == "iw":
            criterion = IW_MaxSquareloss()
            
        if hasattr(criterion, 'requires_reduction') and not criterion.requires_reduction:
            reduction = lambda x, y: x
        else:
            reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        return criterion, reduction

    def run_epoch_pseudo(self, cur_epoch, optimizer, crit, red):
        """
        Run a pseudo-epoch for self-training.

        Args:
            `cur_epoch`: Current epoch index.\n
            `optimizer`: Optimizer object.\n
            `crit`: Criterion object.\n
            `red`: Reduction object.\n

        Returns:
            Early stopping condition (if set) or None
        """
        def pseudo(outs):
            return outs.max(1)[1]
        
        self.model.train()
        seg = self.args.model == "segformer"
        for (images, _) in tqdm(self.train_loader, total=len(self.train_loader)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            images = images.to(self.device, dtype=torch.float32)
            outputs = self._get_outputs(images, _)
            c = crit(outputs, images, seg=seg)
            p = pseudo(outputs)
            loss = red(c, p)
            optimizer.step()
            
        print(f"\tLoss value at epoch {cur_epoch+1}/{self.args.num_epochs_c}: {loss.item()}")
        
        if self.args.es:
            return self.early_stopper.early_stop(loss.item())
        
        return False
    
    def run_epoch(self, cur_epoch, optimizer):
        """
        Run a single epoch of training (on source/centralized dataset).

        Args:
            `cur_epoch`: Current epoch index.

            `optimizer`: Optimizer object.

        Returns:
            None
        """
        self.model.train()
        for (images, labels) in tqdm(self.train_loader, total=len(self.train_loader)):
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            optimizer.zero_grad()
            
            outputs = self._get_outputs(images, labels)
            if self.args.model == "bisenetv2":
                for log in outputs:
                    loss += self.reduction(self.criterion(log,labels),labels)
            else:
                loss = self.reduction(self.criterion(outputs,labels),labels)
            loss.backward()
            # Update parameters
            optimizer.step()
            
        print(f"\tLoss value at epoch {cur_epoch+1}/{self.args.num_epochs}: {loss.item()}")
        
    
    def get_optimizer_and_scheduler(self):
        """
        Get the optimizer and scheduler based on the run configuration.

        Returns:
            Optimizer object and scheduler object.
        """
         # Optimizer chocie
        if self.args.opt == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.m)
        elif self.args.opt == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        else:
            raise NotImplementedError
        
        # Scheduler choice
        if self.args.sched == "lin":
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.args.num_epochs)
        elif self.args.sched == "step":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        else:
            scheduler = None

        return optimizer, scheduler

    def set_set_style_tf_fn(self, styleaug):
        """
        Set the style transfer function for the client's dataset.

        Args:
            `styleaug`: Style augmentation object.

        Returns:
            None
        """
        self.styleaug = styleaug
        self.train_loader.dataset.set_style_tf_fn(self.styleaug.apply_style)

    def train(self, eval_metric=None, eval_datasets=None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        Args:
            `eval_metric`: Evaluation metric object. Default: None
            `eval_datasets`: List of evaluation datasets. Default: None

        Returns:
            Length of the local dataset, model's state dictionary.
        """
        
        optimizer, scheduler = self.get_optimizer_and_scheduler()

        # This flag is used to save the chp with an adequate name 
        is_source = False

        # Section to evalutation at each epoch if --val param is set
        best_miou = 0 if eval_metric else None
        if eval_datasets:
            if len(eval_datasets)>1:
                m = ["same_domain", "diff_domain", "train"]
                eval_datasets.append(self.train_loader)
            else:
                m = ["same_domain"]
        
        # Set model in train mode
        self.model.train()

        stop_condition = False

        # If the teacher is set it means we are in FDA mode
        if self.args.fda:
            if self.teacher:
                crit, red = self.__get_criterion_and_reduction_rules()
            else:
                # This flag is used to save the chp with an adequate name 
                is_source = True
        

        print("-----------------------------------------------------")

        if self.args.num_epochs_c == None:
            self.args.num_epochs_c = 1
        num_epochs = self.args.num_epochs_c if self.teacher else self.args.num_epochs

        for epoch in range(num_epochs):
            # If FDA mode: run epoch with self training 
            if self.teacher:
                self.run_epoch_pseudo(epoch, optimizer, crit, red)
            # Otherwise: standard run
            else:
                self.run_epoch(epoch, optimizer)

            if scheduler:
                scheduler.step()

            # If --val is enabled, evalutes on the train/test sets and save checkpoints if --chp enabled
            if eval_metric and eval_datasets and self.args.val:
                for i, eval_dataset in enumerate(eval_datasets):
                    if i != 0:
                        if not self.args.plot:
                            continue
                    eval_miou=self.test(eval_metric, True, eval_dataset)
                    print(f"\tValidation MioU on {m[i]}: {eval_miou}")
                    self.mious[i].append(eval_miou)
                    if self.args.chp and (eval_miou>best_miou) and i == 0:
                            best_miou = eval_miou
                            torch.save(self.model.state_dict(), f"models/checkpoints/{get_save_string(self.args, is_source)}_checkpoint.pth")
                            print(f"\tSaved checkpoint at epoch {epoch+1}.")
                self.model.train()

                # If early stopping enabled (--es) then check if we have to stop 
                if self.args.es:
                    stop_condition = self.early_stopper.early_stop(eval_miou)
                if(stop_condition):
                    print(f"Training stopped at epoch {epoch+1}: Stopping condition satisfied")
                    break
                
            
        print("-----------------------------------------------------")

        # Save Graph if --val enabled
        if self.args.plot:
            self.plot_loss_miou()

        return len(self.dataset), self.model.state_dict()

    def test(self, metric, eval=None, eval_dataset=None):
        """
        Test the model on the client's dataset.

        Args:
            `metric`: Evaluation metric object.
            `eval`: Flag indicating if evaluation is being performed. Default: None
            `eval_dataset`: Evaluation dataset. Default: None

        Returns:
            Mean IoU score if evaluation flag is set.
        """
        self.model.eval()
        if eval and eval_dataset:
            test_loader = eval_dataset
            metric.reset()
        else:
            test_loader = self.test_loader

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs=self._get_outputs(images, labels, test=True)
                self.update_metric(metric, outputs, labels)
        if eval:
            return metric.get_results()["Mean IoU"]
    
    def plot_loss_miou(self):
        """
        Plot the training mIoU and validation mIoU over epochs.

        This method generates a line chart showing the training mIoU and validation mIoU
        over the epochs. It plots the mIoU values stored in the `mious` attribute of the
        `Client` object.

        The chart is saved as an image file named "miou_vs_miou.png".

        Returns:
            None
        """
        # Sample data
        epochs = range(len(self.mious[0]))
        
        # Create a line chart with two y-values
        plt.plot(epochs, self.mious[2], label='train_miou')
        plt.plot(epochs, self.mious[0], label='val_miou_same')
        plt.plot(epochs, self.mious[1], label='val_miou_diff')


        # Add labels and title
        plt.xlabel('epochs')
        plt.title('Training mioU vs Validation mIoU')

        # Add legend
        plt.legend()

        # Display the chart
        plt.savefig('miou_vs_miou.png')
