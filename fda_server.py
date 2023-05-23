import copy
from collections import OrderedDict

import numpy as np
import torch
from utils.style_transfer import StyleAugment
from PIL import Image
import datasets.ss_transforms as sstr
import matplotlib.pyplot as plt

class FdaServer:
    def __init__(self, args, source_dataset, train_clients, test_clients, model, metrics, valid=False, valid_clients=None):
        self.args = args
        self.source_dataset = source_dataset
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.validation_clients = valid_clients
        self.source_model = model
        self.metrics = metrics
        self.activate_val = valid
        self.model_params_dict = copy.deepcopy(self.source_model.state_dict())

        self.teacher_model = None
        self.student_model = None

        # Style transfer
        self.styleaug = StyleAugment(args.n_images_per_style, args.fda_L, args.fda_size, b=args.fda_b) 
        
        if not args.load:
            self.extract_styles()
        
    def select_clients(self, seed=None):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.args.seed)
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def extract_styles(self):
        for c in self.train_clients:
            self.styleaug.add_style(c.dataset)
    
    def train_source(self):

        if self.args.load:
            pth = "models/checkpoints/source_checkpoint.pth" if self.args.chp else "models/source_best_model.pth"
            saved_params = torch.load(pth)
            self.model_params_dict = saved_params
            self.source_model.load_state_dict(saved_params)
            to_print = " from checkpoints." if self.args.chp else "."
            print(f"Source model loaded{to_print}")
        else:
            _, model_dict = self.train_round_source(self.source_dataset)
            self.model_params_dict = model_dict
            self.source_model.load_state_dict(self.model_params_dict)

        if self.args.save:
                print("Saving training source...")
                torch.save(self.model_params_dict, 'models/source_best_model.pth')

    def train_round_source(self, client):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of one client containing the source dataset to train
            :return: model updates gathered from the clients, to be aggregated
        """
        # Test client augmetation
        print(f"\n Training on source dataset starting.. Num. of samples: {len(client[0].dataset)}")
        #Update parameters of the client model
        client[0].set_set_style_tf_fn(self.styleaug)
        client[0].model.load_state_dict(self.model_params_dict)
        # Temp line. setup train
        update = client[0].train()
        return update
    
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
            student_params = copy.deepcopy(self.student_model.state_dict())
            c.model.load_state_dict(student_params)
            c.set_teacher(self.teacher_model)
            c.early_stopper.reset_counter()
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
        eval_miou_base = 0 
        if self.args.centr:
            num_rounds = 1

        # Centralized train on source dataset
        self.train_source()

        self.test()
        
        # Setup teacher and student
        self.teacher_model = copy.deepcopy(self.source_model)
        self.student_model = copy.deepcopy(self.source_model)

        # Start of distributed train
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

            # Save in the student model the aggregated weights
            self.student_model.load_state_dict(self.model_params_dict)
            
            """if self.activate_val:
                eval_miou=self.eval_validation()
                if self.args.chp and (eval_miou>eval_miou_base):
                    eval_miou_base = eval_miou
                    torch.save(self.source_model.state_dict(), "models/checkpoints/checkpoint.pth")
                    print(f"Changed checkpoint at round {r} with miou:{eval_miou}")

        if self.args.save and (self.args.chp == False):
                print("Saving model...")
                torch.save(self.model_params_dict, 'models/fda_best_model.pth')
    
        if self.args.dataset != "gta5":  
            print("------------------------------------")
            print(f"Evaluation of the trainset started.")
            print("------------------------------------")      
            self.eval_train()"""

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
        self.validation_clients[0].model.load_state_dict(self.student_model.state_dict())
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
        self.metrics["test_same_dom"].reset()
        self.test_clients[0].model.load_state_dict(self.model_params_dict)
        self.test_clients[0].test(self.metrics["test_same_dom"])
        res=self.metrics["test_same_dom"].get_results()
        print(f'Acc: {res["Overall Acc"]}, Mean IoU: {res["Mean IoU"]}')

        print("------------------------------------")
        print(f"Test on DIFFERENT DOMAIN DATA started.")
        print("------------------------------------")
        self.metrics["test_diff_dom"].reset()
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
            self.source_model.eval()

            # Perform inference
            with torch.no_grad():
                output = self.source_model(input_tensor)['out']  # Get the output logits
            output = output.squeeze(0).cpu().numpy()
        
            normalized_output = (output - output.min()) / (output.max() - output.min())

            predicted_labels = np.argmax(normalized_output, axis=0)

            # Normalize the predicted labels to the range [0, 1]
            colormap = plt.cm.get_cmap('tab20', predicted_labels.max() + 1)

            # Create the predicted image with colors
            predicted_image = Image.fromarray((colormap(predicted_labels) * 255).astype(np.uint8))
            
            # Save the predicted image
            class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegatation", "terrain", "sky", "person", "rider", "car", "motorcycle", "bicycle"]
            # Create a legend
            legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colormap(i)) for i in range(len(class_names))]

        # Create a figure and axes
            fig, ax = plt.subplots()

            # Display the predicted image
            ax.imshow(predicted_image)
            ax.axis('off')

            # Create the legend outside the image
            legend = ax.legend(legend_elements, class_names, loc='center left', bbox_to_anchor=(1, 0.5))
            # Adjust the positioning and appearance of the legend
            legend.set_title('Legend')
            frame = legend.get_frame()
            frame.set_edgecolor('black')
            frame.set_facecolor('white')

            # Save the figure
            plt.savefig('fda_image_fin.png', bbox_inches='tight', dpi=300)

            #self.__predict_image()

    def __predict_image(self):
        # Temp function to save image of the style transfer (FDA)
        # Load and preprocess the input image
        dataset = self.source_dataset[0].dataset
        dataset.return_unprocessed_image = True
        input_image = self.styleaug.apply_style(dataset[0])
        # Create the predicted image with colors
    
        fig, ax = plt.subplots()

        # Display the predicted image
        ax.imshow(input_image)
        ax.axis('off')
        # Save the figure
        plt.savefig('fda_transform.png', bbox_inches='tight', dpi=300)


        
