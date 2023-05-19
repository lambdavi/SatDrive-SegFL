import copy
from collections import OrderedDict

import numpy as np
import torch
from utils.style_transfer import StyleAugment

class Server:
    def __init__(self, args, train_clients, test_clients, model, metrics, valid=False, valid_clients=None):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.validation_clients = valid_clients
        self.model = model
        self.metrics = metrics
        self.activate_val = valid
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        # Style transfer
        self.styleaug = StyleAugment(args.n_images_per_style, args.fda_L, args.fda_size, b=args.fda_b)
        self.styleaug.add_style(self.validation_clients[0]) 
        self.styleaug.add_style(self.test_clients[0]) 
        self.styleaug.add_style(self.test_clients[1]) 
        
    def select_clients(self, seed=None):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.args.seed)
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        # Test client augmetation
        
        for i, c in enumerate(clients):
            print(f"Client: {c.name} turn: Num. of samples: {len(c.dataset)}, ({i+1}/{len(clients)})")
            #Update parameters of the client model
            c.set_set_style_tf_fn(self.styleaug)
            c.model.load_state_dict(self.model_params_dict)
            # Temp line. setup train
            update = c.train()
            updates.append(update)
        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        total_weight = 0.
        base = OrderedDict()

        for (client_samples, client_model) in updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)

        averaged_sol_n = copy.deepcopy(self.model_params_dict)

        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to('cuda') / total_weight

        return averaged_sol_n

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        num_rounds = self.args.num_rounds

        if self.args.centr:
            num_rounds = 1
        
        for r in range(num_rounds):
            print("------------------")
            print(f"Round {r+1}/{num_rounds} started.")
            print("------------------")

            # Select random subset of clients
            chosen_client = self.select_clients(seed=r)
            # Train a round
            updates = self.train_round(chosen_client)
            # Aggregate the parameters
            self.model_params_dict = self.aggregate(updates)
            self.model.load_state_dict(self.model_params_dict, strict=False)
            if self.activate_val:
                self.eval_validation()

        
        if self.args.dataset != "gta5":  
            print("------------------------------------")
            print(f"Evaluation of the trainset started.")
            print("------------------------------------")      
            self.eval_train()
        self.test()

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        self.metrics["eval_train"].reset()
        for c in self.train_clients:
            c.model.load_state_dict(self.model_params_dict)
            c.test(self.metrics["eval_train"])
        res=self.metrics["eval_train"].get_results()
        print(f'Acc: {res["Overall Acc"]}, Mean IoU: {res["Mean IoU"]}')

    def eval_validation(self):
        """
        This method handles the evaluation on the validation client(s)
        """
        self.metrics["eval_train"].reset()
        self.validation_clients[0].model.load_state_dict(self.model_params_dict)
        self.validation_clients[0].test(self.metrics["eval_train"])
        res=self.metrics["eval_train"].get_results()
        print(f'Validation: Mean IoU: {res["Mean IoU"]}')

    def test(self):
        """
            This method handles the test on the test clients
        """
        print("------------------------------------")
        print(f"Test on SAME DOMAIN DATA started.")
        print("------------------------------------")
        self.test_clients[0].model.load_state_dict(self.model_params_dict)
        self.test_clients[0].test(self.metrics["test_same_dom"])
        res=self.metrics["test_same_dom"].get_results()
        print(f'Acc: {res["Overall Acc"]}, Mean IoU: {res["Mean IoU"]}')
        print("------------------------------------")
        print(f"Test on DIFFERENT DOMAIN DATA started.")
        print("------------------------------------")
        self.test_clients[1].model.load_state_dict(self.model_params_dict)
        self.test_clients[1].test(self.metrics["test_diff_dom"])
        res=self.metrics["test_diff_dom"].get_results()
        print(f'Acc: {res["Overall Acc"]}, Mean IoU: {res["Mean IoU"]}')

        
