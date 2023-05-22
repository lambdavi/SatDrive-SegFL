# Real World Federated Learning
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
Code for the Federated Learning project.

#### Datasets
The repository supports experiments on the following datasets:
1. Reduced **Federated IDDA** from FedDrive [2]
   - Task: semantic segmentation for autonomous driving
   - 24 users
2. Reduced **GTA5**
   - Task: semantic segmentation for autonomous driving
   - downloadGta.py available to download it.

## How to run
The ```main.py``` orchestrates training. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```).
Example of experiments:

### Centralized mode: 
- **IDDA** (Semantic Segmentation)
```bash
python main.py --dataset idda --centr --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 20 --clients_per_round 8 
```

### Distributed mode: 
- **GTA** (Semantic Segmentation)
```bash
python main.py --dataset gta5 --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 2 --clients_per_round 8 
```

### FDA mode: 
- **GTA** (Semantic Segmentation)
```bash
python main.py --dataset idda --fda --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 2 --clients_per_round 8 
```

## References
[1] Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." Workshop on Federated Learning for Data Privacy and Confidentiality (2019). 

[2] Fantauzzo, Lidia, et al. "FedDrive: generalizing federated learning to semantic segmentation in autonomous driving." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.
