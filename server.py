import copy
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
import datasets.ss_transforms as sstr
import matplotlib.pyplot as plt
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
        for i, c in enumerate(clients):
            print(f"Client: {c.name} turn: Num. of samples: {len(c.dataset)}, ({i+1}/{len(clients)})")
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

        num_rounds = self.args.num_rounds
        eval_miou_base = 0  # used to save checkpoints if needed
        if self.args.centr:
            num_rounds = 1
        
        if self.args.load:
            pth = "models/checkpoints/checkpoint.pth" if self.args.chp else "models/best_model.pth"
            saved_params = torch.load(pth)
            self.model_params_dict = saved_params
            self.model.load_state_dict(saved_params)
            self.model.eval()
            print("Model Loaded!")
        else:
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
                    eval_miou=self.eval_validation()
                    if self.args.chp and (eval_miou>eval_miou_base):
                        eval_miou_base = eval_miou
                        torch.save(self.model.state_dict(), "models/checkpoints/checkpoint.pth")
                        print(f"Changed checkpoint at round {r} with miou:{eval_miou}")
            
            if self.args.save and (self.args.chp == False):
                print("Saving model...")
                torch.save(self.model_params_dict, 'models/best_model.pth')

        if self.args.dataset != "gta5":  
            print("------------------------------------")
            print(f"Evaluation of the trainset started.")
            print("------------------------------------")      
            #self.eval_train()
        #self.test()

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
        return res["Mean IoU"]

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

    def predict(self, image_path):

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
            output = self.model(input_tensor)['out']  # Get the output logits
        output = output.squeeze(0).cpu().numpy()
        print(output.shape)
        print(output[0][0].shape)
        print(np.unique(output[0][0]))
        normalized_output = (output - output.min()) / (output.max() - output.min())


        predicted_labels = np.argmax(normalized_output, axis=0)
        #predicted_labels = np.argmax(normalized_output, axis=0).astype(np.uint8)

        print(predicted_labels.shape)

        # Normalize the predicted labels to the range [0, 1]
        colormap = plt.cm.get_cmap('tab20', predicted_labels.max() + 1)

        # Create the predicted image with colors
        predicted_image = Image.fromarray((colormap(predicted_labels) * 255).astype(np.uint8))
        
        # Save the predicted image
        predicted_image.save('test_semantic.png')
        class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegatation", "terrain", "sky", "person", "rider", "car", "motorcycle", "bicycle"]
        # Create a legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colormap(i)) for i in range(len(class_names))]

        # Plot the legend
        plt.figure()
        plt.imshow(predicted_image)
        plt.legend(legend_elements, class_names)
        plt.axis('off')
        plt.savefig('image_fin.png')



