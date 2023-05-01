import copy
from collections import OrderedDict

import numpy as np
import torch


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        np.random.seed(self.args.seed)
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, c in enumerate(clients):
            print(f"Client: {c.name} turn. ({i+1}/{len(clients)})")
            #Update parameters of the client model
            c.model.load_state_dict(self.model_params_dict)
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
        for r in range(self.args.num_rounds):
            print("------------------")
            print(f"Round {r} started.")
            print("------------------")

            # Select random subset of clients
            chosen_client = self.select_clients()
            # Train a round
            updates = self.train_round(chosen_client)
            # Aggregate the parameters
            self.model_params_dict = self.aggregate(updates)
            self.model.load_state_dict(self.model_params_dict, strict=False)            
            ## self.eval_train()

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        # TODO: missing code here!
        raise NotImplementedError

    def test(self):
        """
            This method handles the test on the test clients
        """
        # TODO: missing code here!
        raise NotImplementedError
