# Exploring Federated Learning for Semantic Segmentation in Autonomous Driving Scenarios
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
Code for the "Federated Learning for Autonomous Driving" project.

#### Datasets
The repository supports experiments on the following datasets:
1. Reduced **Federated IDDA** from FedDrive [2]
   - Task: semantic segmentation for autonomous driving
   - 24 users
2. Reduced **GTA5**
   - Task: semantic segmentation for autonomous driving
   - downloadGta.py available to download it.
2. Reduced **LoveDA**
   - Task: semantic segmentation for satellite/aerial imagery.
   - downloadLoveda.py available to download it.

NOTE: to use the scripts and download the files run before (if needed): ```pip install pymegatools```

## How to run
The ```main.py``` setup the whole application. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```).
Example of experiments:

### Centralized mode: 
- **IDDA** 
```bash
python main.py --dataset idda --centr --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 20 --clients_per_round 8 
```

### Distributed mode: 
- **GTA5** 
```bash
python main.py --dataset idda --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 2 --clients_per_round 8 
```

### FDA mode (pretraining + distributed mode): 
- **GTA as source and IDDA as target** 
```bash
python main.py --dataset idda --fda --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 2 --clients_per_round 8
```

## Reproducing results:
Some checkpoints are already available in the repo. In particular they are located in ```models/checkpoints```.

To reproduce the result you can use the args --load_from (e.g.):

- **GTA as source and IDDA as target (FDA + Pseudo)** 
```bash
python main.py --dataset idda --fda --model deeplabv3_mobilenetv2 --load_from "models/checkpoints/gta5_fda.pth" --num_rounds 200 --num_epochs 2 --clients_per_round 8
```
- **Loveda (Pseudo)** 
```bash
python main.py --dataset loveda --fda --model deeplabv3_mobilenetv2 --load_from "models/checkpoints/loveda_nofda.pth" --num_rounds 200 --num_epochs 2 --clients_per_round 8
```
- **GTA5 CENTR TRANSFORMER** 
```bash
python main.py --dataset gta5 --model segformer --transformer_model b1 --load_from "models/checkpoints/gta5_nofda.pth" --num_rounds 1 --num_epochs 1 --clients_per_round 1
```

## Checkpoints available:
### gta5_fda
Load the best checkpoint of the model trained on gta5 dataset + FDA transformation.

### gta5_nofda
Load the best checkpoint of the model trained on gta5 dataset w/o FDA.

### loveda_nofda.pth
Load the best checkpoint of the model trained loveda.

All the checkpoints are saved on a run of 100 epochs.

For other checkpoints (e.g. Segformer checkpoints):
[Drive Checkpoints](https://drive.google.com/drive/folders/1tN2UJx91axP7mkj51X1SL3WceAXCCCGv?usp=sharing)



