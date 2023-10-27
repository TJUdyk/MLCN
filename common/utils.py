import os
import wandb
import torch
import pprint
import random
import cv2
import time
import argparse
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from termcolor import colored


def setup_run(arg_mode='train'):
    args = parse_args(arg_mode=arg_mode)

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'checkpoints/{args.dataset}/{args.shot}shot-{args.way}way/', args.extra_dir)
    ensure_path(args.save_path)

    if not args.no_wandb:
        wandb.init(project=f'mlcn-{args.dataset}-{args.way}w{args.shot}s',
                   config=args,
                   save_code=True,
                   name=args.extra_dir)

    if args.dataset == 'miniimagenet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'fc100':
        args.num_class = 60
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
    elif args.dataset == 'cifar_fs':
        args.num_class = 64
    elif args.dataset == 'cars':
        args.num_class = 130
    elif args.dataset == 'dogs':
        args.num_class = 70

    return args


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


def failure_analysis(args , data , logits , labels , classes):
    path = os.path.join(f'datasets/wrong/{args.dataset}/{args.shot}shot-{args.way}way/',args.extra_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    
    failure_index = logits.argmax(dim=1) != labels
    failure_data = data[failure_index]
    failure_label = logits.argmax(dim=1)[failure_index]
    true_label = labels[failure_index]

    failure_classes = []
    true_classes = []
    for label1, label2 in zip(failure_label.cpu().numpy(), true_label.cpu().numpy()):
        failure_classes.append(classes[label1])
        true_classes.append(classes[label2])

    for data, label1, label2 in zip(failure_data, true_classes, failure_classes):
        data = data.cpu().numpy().transpose(1, 2, 0)
        data = data * 255
        data = data.astype(np.uint8)
        timestamp = int(time.time() * 100000)
        cv2.imwrite(os.path.join(path, "{}-{}-{}.jpg".format(label1, label2, timestamp)), data[:, :, ::-1])


def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def load_model(model, dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(dir)['params']

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:
        print('loading model from :', dir)
        if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
            if 'module' in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
        model.load_state_dict(model_dict)

    return model

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cosine_similarity(x, y):
    # x: N x D
    # y: M x D
    cos = nn.CosineSimilarity(dim=0)
    cos_sim = []
    for xi in x:
        cos_sim_i = []
        for yj in y:
            cos_sim_i.append(cos(xi, yj))
        cos_sim_i = torch.stack(cos_sim_i)
        cos_sim.append(cos_sim_i)
    cos_sim = torch.stack(cos_sim)
    return cos_sim  # (N, M)

def euclidean_dist(x, y):
    # x :  torch.Size([5, 640]) y : torch.Size([75, 640]) 
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def euclidean_dist_similarity(x, y):

    return -torch.pow(x - y, 2).sum(2)  # N*M


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def detect_grad_nan(model):
    for param in model.parameters():
        if param.grad == None:
            pass
        elif (param.grad != param.grad).float().sum() != 0: 
            param.grad.zero_()


def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow


def parse_args(arg_mode):
    parser = argparse.ArgumentParser(description='')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='cub',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'cifar_fs'])
    parser.add_argument('-data_dir', type=str, default='datasets', help='dir of datasets')  
    parser.add_argument('-batch', type=int, default=64, help='auxiliary batch size')

    ''' about training schedules '''
    parser.add_argument('-max_epoch', type=int, default=100, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')

    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[60 , 80 ], help='milestones for MultiStepLR')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('--learning_rate', type=float, default=5e-2, help='learning rate')


    parser.add_argument('-temperature_attn', type=float, default=5.0, metavar='gamma', help='temperature for softmax in computing cross-attention')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')

    ''' about metric  '''
    parser.add_argument('-lamb', type=float, default=1.5, metavar='lambda', help='loss balancing term')
    parser.add_argument('-temperature', type=float, default=0.1, metavar='tau', help='temperature for metric-based loss')

    ''' about env '''
    parser.add_argument('-gpu', default='1', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-extra_dir', type=str, default='test222', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-no_wandb', action='store_true', help='not plotting learning curve on wandb',
                        default=arg_mode == 'test')  # train: enable logging / test: disable logging

    args = parser.parse_args()
    return args
