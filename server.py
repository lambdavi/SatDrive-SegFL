import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import datasets.ss_transforms as sstr
from utils.utils import get_save_string


class Server:
    """
    A class representing the Server for both centralized mode and distributed mode without FDA.

    The Server is responsible for coordinating the training process of the system. It handles
    the client-server framework.

    Args:
        args: An object containing the arguments and configuration for the FDA server.
        train_clients: A list of training clients.
        test_clients: A list of test clients.
        model: The initial source model.
        metrics: A dictionary containing evaluation metrics.
        valid (bool): Whether to include validation during training. Default is False.
        valid_clients: A list of validation clients. Required if `valid` is True. Default is None.
    """
    def __init__(self, args, train_clients, test_clients, model, metrics, valid=False, valid_clients=None):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.validation_clients = valid_clients
        self.model = model
        self.metrics = metrics
        self.activate_val = valid
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

    def select_clients(self, seed=None):
        """
        Selects a subset of clients for a training round.

        Args:
            seed (int): Random seed for client selection. Default is None.

        Returns:
            numpy.ndarray: An array of selected clients for the training round.
        """
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.args.seed)
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            Args:
                `clients`: list of all the clients to train
            Returns:
                list[tuple]: [(len_dataset, state_dictionary)]. Model updates gathered from the client to be aggregated.
        """
        updates = []

        for i, c in enumerate(clients):
            print(f"Client: {c.name} turn: Num. of samples: {len(c.dataset)}, ({i+1}/{len(clients)})")

            # Update parameters of the client model
            c.model.load_state_dict(self.model_params_dict)

            if self.args.val:
                self.metrics["eval_train"].reset()
                update = c.train(self.metrics["eval_train"], [self.test_clients[0].test_loader, self.test_clients[1].test_loader]) 
            else:
                update = c.train()    

            updates.append(update)
            
        return updates

    def aggregate(self, updates):
        """
        Aggregates the model updates received from the clients using FedAvg.

        Args:
            `updates`: Model updates received from the clients. List of tuples.

        Returns:
            OrderedDict: Aggregated model parameters.
        """
        total_weight = 0.
        base = OrderedDict()

        # Numerator of weighted average
        for (client_samples, client_model) in updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)

        averaged_sol_n = copy.deepcopy(self.model_params_dict)

        # Denominator of weighted average
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to('cuda') / total_weight

        # Return weighted average
        return averaged_sol_n

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        retrain_error=False
        num_rounds = self.args.num_rounds
        if self.args.centr:
            num_rounds = 1
        
        if self.args.load or self.args.resume:
            pth = f"models/checkpoints/{get_save_string(self.args, False)}_checkpoint.pth" if self.args.chp else f"models/{get_save_string(self.args, False)}_best_model.pth"
            try:
                saved_params = torch.load(pth)
                self.model_params_dict = saved_params
                self.model.load_state_dict(saved_params)
                self.model.eval()
                to_print = " from checkpoints." if self.args.chp else "."
                print(f"Model loaded{to_print}")
            except:
                print("The checkpoint for this model does not exist. The model will be retrained.")
                retrain_error=True
            
        if (not self.args.load) or self.args.resume or retrain_error:
            for r in range(num_rounds):
                print("------------------")
                print(f"Round {r+1}/{num_rounds} started.")
                print("------------------")

                # Select random subset of clients
                chosen_clients = self.select_clients(seed=r)
                
                # Train a round
                updates = self.train_round(chosen_clients)

                # Aggregate the parameters
                self.model_params_dict = self.aggregate(updates)
                self.model.load_state_dict(self.model_params_dict, strict=False) 
            
            if self.args.save:
                print("Saving model...")
                torch.save(self.model_params_dict, f'models/{get_save_string(self.args, False)}_best_model.pth')

              
            self.eval_train()
            if self.validation_clients:
                self.eval_validation()
            self.test()

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        print("------------------------------------")
        print(f"Evaluation of the trainset started.")
        print("------------------------------------") 

        self.metrics["eval_train"].reset()

        for c in self.train_clients:
            c.model.load_state_dict((copy.deepcopy(self.model_params_dict)))
            c.test(self.metrics["eval_train"])

        res=self.metrics["eval_train"].get_results()
        print(f'Acc: {res["Overall Acc"]}, Mean IoU: {res["Mean IoU"]}')

    def eval_validation(self):
        """
        This method handles the evaluation on the validation client(s)
        """
        print("------------------------------------")
        print(f"Evaluation of the validation set started.")
        print("------------------------------------") 

        self.metrics["eval_train"].reset()

        self.validation_clients[0].model.load_state_dict((copy.deepcopy(self.model_params_dict)))
        self.validation_clients[0].test(self.metrics["eval_train"])
    
        res=self.metrics["eval_train"].get_results()
        print(f'Validation: Mean IoU: {res["Mean IoU"]}')
        return res["Mean IoU"]

    def test(self):
        """
            This method handles the test on the test clients
        """
        print("------------------------------------")
        print(f"Test on SAME DOMAIN DATA started.")
        print("------------------------------------")

        self.test_clients[0].model.load_state_dict((copy.deepcopy(self.model_params_dict)))
        self.test_clients[0].test(self.metrics["test_same_dom"])

        res=self.metrics["test_same_dom"].get_results()
        print(f'Acc: {res["Overall Acc"]}, Mean IoU: {res["Mean IoU"]}')

        print("------------------------------------")
        print(f"Test on DIFFERENT DOMAIN DATA started.")
        print("------------------------------------")

        self.test_clients[1].model.load_state_dict((copy.deepcopy(self.model_params_dict)))
        self.test_clients[1].test(self.metrics["test_diff_dom"])

        res=self.metrics["test_diff_dom"].get_results()
        print(f'Acc: {res["Overall Acc"]}, Mean IoU: {res["Mean IoU"]}')

    def predict(self, image_path):
        """
        Handles the the prediction. Outputs an image in the root directory.

        Args: 
            `image_path`: path to the image to predict.
        """
        def get_outputs(images, labels=None, test=False):
            if self.args.model == 'deeplabv3_mobilenetv2':
                return self.model(images)['out']
            if self.args.model in ['resnet18',]:
                return self.model(images)
            if self.args.model == 'segformer':
                logits = self.model(images).logits
                outputs = torch.nn.functional.interpolate(
                        logits, 
                        size=images.shape[-2:], 
                        mode="bilinear", 
                        align_corners=False
                )
                return outputs
            if self.args.model == 'bisenetv2':
                outputs = self.model(images, test=test)
                return outputs
        # Load and preprocess the input image
        input_image = Image.open(image_path)

        # Apply necessary transformations
        transforms= sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transforms(input_image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.cuda()
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            output = get_outputs(input_tensor, test=True)  # Get the output logits
        output = output.squeeze(0).cpu().numpy()
    
        normalized_output = (output - output.min()) / (output.max() - output.min())

        predicted_labels = np.argmax(normalized_output, axis=0)

        colormap = plt.cm.get_cmap('tab20', predicted_labels.max() + 1)

        # Create the predicted image with colors
        predicted_image = Image.fromarray((colormap(predicted_labels) * 255).astype(np.uint8))
        
        # Save the predicted image
        if self.args.dataset != "loveda":
            class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegatation", "terrain", "sky", "person", "rider", "car", "motorcycle", "bicycle"]
        else:
            class_names = ["DC", "background", "building", "road", "water", "barren", "forest", "agriculture"]

        # Create a legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colormap(i)) for i in range(len(class_names))]

        # Create a figure and axes
        _, ax = plt.subplots()

        # Display the predicted image
        ax.imshow(np.array(input_image))

        if self.args.dataset == "loveda":
            ax.imshow(predicted_image, alpha=0.4)
        else:  
            ax.imshow(predicted_image, alpha=0.7)
        ax.axis('off')

        # Create the legend outside the image
        legend = ax.legend(legend_elements, class_names, loc='center left', bbox_to_anchor=(1, 0.5))
        # Adjust the positioning and appearance of the legend
        legend.set_title('Legend')
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_facecolor('white')

        # Save the figure
        plt.savefig('image_fin.png', bbox_inches='tight', dpi=300)



