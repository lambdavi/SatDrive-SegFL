

### General information
The driver of the code is the *main.py* file.
The most important (and some required) arguments are:
--dataset {'idda', 'gta5', 'loveda'}
--model {'deeplabv3_mobilenetv2', 'resnet18', 'segformer', 'bisenetv2'}
--num_epochs NUM_EPOCHS

To select the mode:
--centr: sets the centralized mode. The model will be trained only on the selected dataset and evaluted on the dataset own test sets.
    (For GTA, IDDA test datasets are used).
--fda: sets FDA mode. The model will be pretrained on the source dataset and then the distributed mode will be activated and the clients    will use the teacher-student framework for the self-training. If the inrest is only in the centralized part (pretraining), just ignore the second part of the code.
    Two options are explored and allowed.
    If dataset=idda -> idda will be used as clients and GTA for the pretraining phase.
    If dataset=loveda -> loveda (on urban data) will be used for pretraining and loveda with both urban and rural data for the next phase.

If none of this mode is set, the "distributed" framework (client-server) is used.

Commands like:
--save, --chp, --resume and --load can be used to manage checkpoints and saved models.

To see more about arguments see in utils/args.py

## 