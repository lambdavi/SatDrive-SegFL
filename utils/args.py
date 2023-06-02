import argparse

INF = 9999
def str2tuple(tp=int):

    def convert(s):
        return tuple(tp(i) for i in s.split(','))
    return convert


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist', 'gta5', 'loveda'], required=True, help='dataset name')
    parser.add_argument('--niid', action='store_true', default=False,  help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'segformer', 'bisenetv2'], help='model name')
    parser.add_argument('--transformer_model', type=str, choices=['b0', 'b1', 'b2'], default="b0", help='weights of the transformers b0 lighter - b2 heavier')
    parser.add_argument('--num_rounds', type=int, default=1, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, default=1, help='number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')
    # New Argument:s
    parser.add_argument('--centr', action='store_true', default=False, help='Only one client will be used if set True')
    parser.add_argument('--fda', action='store_true', default=False, help='FDA mode activated')
    parser.add_argument('--opt', type=str, choices=['SGD', 'adam'], default = 'SGD', help='Optimizer choice')
    parser.add_argument('--sched', type=str, choices=['lin', 'step'], default = None, help='Scheduler choice')
    parser.add_argument('--n_images_per_style', type=int, default=1000, help='number of images to extract style (avg is performed)')
    parser.add_argument('--fda_L', type=float, default=0.01, help='to control size of amplitude window')
    parser.add_argument('--fda_b', type=int, default=None, help='if != None it is used instead of fda_L:' 'b == 0 --> 1x1, b == 1 --> 3x3, b == 2 --> 5x5, ...')
    parser.add_argument('--fda_size', type=str2tuple(int), default='1024,512', help='size (W,H) to which resize images before style transfer')
    parser.add_argument('--es', type=str2tuple(float), default=None, help='patience,tol')
    parser.add_argument('--save', action='store_true', default=False, help='Model saved at the end (training performed)')
    parser.add_argument('--load', action='store_true', default=False, help='Load saved model')
    parser.add_argument('--chp', action='store_true', default=False, help='Model checkpoints saved during training')
    parser.add_argument('--pred', type=str, default = None, help='Path of image to predict')
    parser.add_argument('--loss', type=str, choices=['self', 'iw'], default = "self", help='Loss choice')
    parser.add_argument('--val', action='store_true', default=False, help='Activate validation during training')
    parser.add_argument('--teacher_step', type=int, default = INF, help='How often change the teacher model in FDA settings')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from training model')

    return parser
