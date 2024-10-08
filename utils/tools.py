import numpy as np
import torch
import random
import errno
import os
import sys
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset

from utils.cutmix import  CutMix
from utils.random_crop import RandomCrop
from utils.random_erasing import RandomErasing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_save_path(father_path, args):
    father_path = os.path.join(father_path, '{}'.format(time.strftime("%m_%d_%H_%M")))
    mkdir(father_path)
    args.log_path = father_path
    args.model_path = father_path
    args.result_path = father_path
    args.spatial_adj_path = father_path
    args.time_adj_path = father_path
    args.tensorboard_path = father_path
    return args


def sliding_window_eeg(data,label,window_size,stride):
    # print(data.shape)
    trails = data.shape[0]
    num_channels = data.shape[1]
    num_samples = data.shape[2]
    num_segments = (num_samples - window_size)//stride + 1
    segments = np.zeros((num_segments,trails,num_channels,window_size),dtype=np.float64)

    for i in range(trails):
        for j in range(num_segments):
            start = j * stride
            end = start + window_size
            segments[j][i] = data[i,:,start:end]
    
    return segments,label



EOS = 1e-10
def normalize(adj):
    adj = F.relu(adj)
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sum(adj,dim=-1,keepdim=False)) + EOS)
    return inv_sqrt_degree[:,None]*adj*inv_sqrt_degree[None,:]




def save(checkpoints, save_path):
    torch.save(checkpoints, save_path)


def load_adj(dn='bciciv2a', norm=False):
    if 'hgd' == dn:
        num_node = 44
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 21), (1, 11), (1, 25), (1, 14),
                         (2, 22), (2, 37), (2, 11), (2, 12), (2, 26), (2, 15), (2, 39),
                         (3, 38), (3, 23), (3, 12), (3, 13), (3, 40), (3, 16), (3, 27),
                         (4, 24), (4, 13), (4, 28), (4, 17),
                         (5, 25), (5, 11), (5, 26), (5, 14), (5, 15), (5, 29), (5, 18), (5, 30),
                         (6, 27), (6, 13), (6, 28), (6, 16), (6, 17), (6, 31), (6, 20), (6, 32),
                         (7, 14), (7, 29), (7, 18), (7, 33),
                         (8, 30), (8, 15), (8, 41), (8, 18), (8, 19), (8, 34), (8, 43),
                         (9, 42), (9, 16), (9, 31), (9, 19), (9, 20), (9, 44), (9, 35),
                         (10, 17), (10, 32), (10, 20), (10, 36),
                         (11, 21), (11, 22), (11, 25), (11, 26),
                         (12, 37), (12, 38), (12, 39), (12, 40),
                         (13, 23), (13, 24), (13, 27), (13, 28),
                         (14, 25), (14, 29),
                         (15, 26), (15, 39), (15, 30), (15, 41),
                         (16, 40), (16, 27), (16, 42), (16, 31),
                         (17, 28), (17, 32),
                         (18, 29), (18, 30), (18, 33), (18, 34),
                         (19, 41), (19, 42), (19, 43), (19, 44),
                         (20, 31), (20, 32), (20, 35), (20, 36),
                         (21, 22), (21, 25),
                         (22, 37), (22, 26),
                         (23, 38), (23, 24), (23, 27),
                         (24, 28),
                         (25, 26), (25, 29),
                         (26, 39), (26, 30),
                         (27, 40), (27, 28), (27, 31),
                         (28, 32),
                         (29, 30), (29, 33),
                         (30, 41), (30, 34),
                         (31, 42), (31, 32), (31, 35),
                         (32, 36),
                         (33, 34),
                         (34, 43),
                         (35, 36), (35, 44),
                         (37, 38), (37, 39),
                         (38, 40),
                         (39, 40), (39, 41),
                         (40, 42),
                         (41, 43), (41, 42),
                         (42, 44),
                         (43, 44)]
    elif 'bciciv2a' == dn:
        num_node = 22
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 3), (1, 4), (1, 5),
                         (2, 3), (2, 7), (2, 8), (2, 9),
                         (3, 4), (3, 8), (3, 9), (3, 10),
                         (4, 5), (4, 9), (4, 10), (4, 11),
                         (5, 6), (5, 10), (5, 11), (5, 12),
                         (6, 11), (6, 12), (6, 13),
                         (7, 8), (7, 14),
                         (8, 9), (8, 14), (8, 15),
                         (9, 10), (9, 14), (9, 15), (9, 16),
                         (10, 11), (10, 15), (10, 16), (10, 17),
                         (11, 12), (11, 16), (11, 17), (11, 18),
                         (12, 13), (12, 17), (12, 18),
                         (13, 18),
                         (14, 15), (14, 19),
                         (15, 16), (15, 19), (15, 20),
                         (16, 17), (16, 19), (16, 20), (16, 21),
                         (17, 18), (17, 20), (17, 21),
                         (18, 21),
                         (19, 20), (19, 22),
                         (20, 21), (20, 22),
                         (21, 22)]
    elif 'physionet' == dn:
        num_node = 64
        self_link = [(i, i) for i in range(num_node)]        
        neighbor_link =  [(1, 2), (1,31),(1,39),(1,8),(1,30),(1,32),(1,9),(1,41),
                           (2, 3), (2,32),(2,9),(2,31),(2,33),(2,10),(2,8),
                           (3, 4), (3,33),(3,10),(3,32),(3,34),(3,11),(3,9),
                           (4, 5), (4,34),(4,11),(4,33),(4,45),(4,12),(4,10),
                           (5, 6), (5,35),(5,12),(5,34),(5,36),(5,13),(5,11),
                           (6, 7), (6,36),(6,13),(6,35),(6,37),(6,14),(6,12),
                           (7, 40),(7,37),(7,14),(7,36),(7,38),(7,42),(7,13),
                           (8, 9), (8,15),(8,41),(8,39),(8,16),(8,45),
                           (9, 10), (9,16),(9,15),(9,17),
                           (10, 11), (10,17),(10,16),(10,18),
                           (11, 12), (11,18),(11,17),(11,19),
                           (12, 13), (12,19),(12,18),(12,20),
                           (13, 14), (13,20),(13,19),(13,21),
                           (14, 21), (14,42),(14,40),(14,46),(14,20),
                           (15, 16), (15,45),(15,48),(15,41),(15,47),(15,49),
                           (16, 17), (16,49),(16,48),(16,50),
                           (17, 18), (17,50),(17,51),(17,49),
                           (18, 19), (18,51),(18,50),(18,52),
                           (19, 20), (19,52),(19,51),(19,53),
                           (20, 21), (20,53),(20,52),(20,54),
                           (21, 46), (21,54),(21,42),(21,53),(21,55),
                           (22, 23), (22,26),(22,25),(22,27),
                           (23, 24), (23,27),(23,26),(23,28),
                           (24, 28), (24,27),(24,29),
                           (25, 26), (25,32),(25,31),(25,33),
                           (26, 37), (26,33),(26,34),(26,32),
                           (27, 28), (27,34),(27,33),(27,35),
                           (28, 29), (28,35),(28,34),(28,36),
                           (29, 36), (29,35),(29,37),
                           (30, 31), (30,39),
                           (31, 32), (31,39),
                           (33, 34),
                           (34, 35),
                           (35, 36),
                           (36, 37),
                           (37, 38), (37,40),
                           (38, 40),
                           (39, 41), (39,43),
                           (40, 42), (40,44),
                           (41, 43), (41,45),
                           (42, 44), (42,46),
                           (43, 45),
                           (44, 46),
                           (45, 47), (45,48),
                           (46, 55), (46,54),
                           (47, 48),
                           (48, 49), (48,56),
                           (49, 50), (49,56),(49,57),
                           (50, 51), (50,57),(50,56),(50,58),
                           (51, 52), (51,58),(51,57),(51,59),
                           (52, 53), (52,59),(52,58),(52,60),
                           (53, 54), (53,60),(53,59),
                           (54, 55), (54,60),
                           (56, 57), (56,61),
                           (57, 58), (57,61),(57,62),
                           (58, 59), (58,62),(58,61),(58,63),
                           (59, 60), (59,63),(59,62),
                           (60, 63), 
                           (61, 62), (61,64),
                           (62, 63), (62,64),
                           (63, 64)]
    else:
        raise ValueError('cant support {} dataset'.format(dn))
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
    edge = self_link + neighbor_link
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1.
        A[j, i] = 1.
    if 'physionet' == dn:
        for i in range(64):
            for j in range(64):
                if A[i,j] == 0:
                    A[i,j] = 0.01     
    elif 'bciciv2a' == dn:    
        for i in range(22):
            for j in range(22):
                if A[i,j] == 0:
                    A[i,j] = 0.1
    return A


def accuracy(output, target, topk=(1,)):
    shape = None
    if 2 == len(target.size()):
        shape = target.size()
        target = target.view(target.size(0))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    if shape:
        target = target.view(shape)
    return ret


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, *args):
        for t in self.transforms:
            x = t(x, *args)
        return x

def build_tranforms():
    return Compose([
        RandomCrop(1125),
        CutMix(),  
        # RandomErasing(),
    ])

class EEGDataSet(Dataset):
    def __init__(self,data,label):
        self.label = label
        self.data = data
    
    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, index):
        data = self.data[:,index]
        label = self.label[index]
        return data,label

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


