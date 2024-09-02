import torch
import os
import sys
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter

from utils.tools import set_seed,set_save_path,Logger,sliding_window_eeg,load_adj, build_tranforms,EEGDataSet,save
from utils.dataload import load_bciciv2a_data_single_subject,load_physionet_data_single_subject,load_bci2b_data_single_subject, load_labdata_data_single_subject
from models.STGENet import STGENET
from utils.RepeatedTrialAugmentation import RepeatedTrialAugmentation
from utils.run_epoch import train_one_epoch,evaluate_one_epoch

import matplotlib.pyplot as plt
import pandas as pd

import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader,TensorDataset,Dataset



def start_run(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, f'information-{args.id}.txt'))
    tensorboard = SummaryWriter(args.tensorboard_path)

    start_epoch = 0
    # ------------------------------------------------device setting--------------------------------------------------
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------data setting----------------------------------------------------
    
    if args.data_type == 'bci2a':
        train_X, train_y, test_X, test_y = load_bciciv2a_data_single_subject(args.data_path,subject_id=args.id)
    elif args.data_type == 'physionet':
        train_X, train_y, test_X, test_y = load_physionet_data_single_subject(args.data_path,subject_id=args.id)
    elif args.data_type == 'bci2b':
        # args.data_path = ''
        args.channel_num = 3
        args.n_class = 2
        args.out_chans = 8
        train_X, train_y, test_X, test_y = load_bci2b_data_single_subject(args.data_path,subject_id=args.id)
    elif args.data_type == 'labdata':
        # args.data_path = ''
        args.channel_num = 64
        args.n_class = 4
        train_X, train_y, test_X, test_y = load_labdata_data_single_subject(args.data_path,subject_id=args.id)
    channel_num = args.channel_num
    
    slide_window_length = args.window_length
    slide_window_stride = args.window_padding


    slide_train_X,slide_train_y = sliding_window_eeg(train_X,train_y,slide_window_length,slide_window_stride)
    slide_test_X,slide_test_y = sliding_window_eeg(test_X,test_y,slide_window_length,slide_window_stride)

    # print('dsgvnhiuodsfbuiodfiodsfioarsduraseduhaeruiop',slide_train_X.shape)

    slide_train_X = torch.tensor(slide_train_X, dtype=torch.float32)
    slide_test_X = torch.tensor(slide_test_X, dtype=torch.float32)
    slide_train_y = torch.tensor(slide_train_y, dtype=torch.int64)
    slide_test_y = torch.tensor(slide_test_y, dtype=torch.int64)

    print(slide_train_X.shape,slide_train_y.shape)
    print(slide_test_X.shape,slide_test_y.shape)
    print(train_X.shape,train_y.shape)
    print(test_X.shape,test_y.shape)

    slide_window_num = slide_train_X.shape[0]

    # -----------------------------------------------training setting-------------------------------------------------
    if 'l' == args.spatial_adj_mode:
        # Adj = torch.tensor(load_adj('bciciv2a'), dtype=torch.float32)
        Adj = torch.tensor(load_adj('physionet'),dtype=torch.float32)
        # Adj = torch.tensor(torch.ones(3,3))
    elif 'p' == args.spatial_adj_mode:
        temp = train_X
        train_data = temp.permute(0, 2, 1).contiguous().reshape(-1, channel_num)
        Adj = torch.tensor(np.corrcoef(train_data.numpy().T, ddof=1), dtype=torch.float32)
    elif 'r' == args.spatial_adj_mode:
        Adj = torch.randn(channel_num,channel_num)
    elif 'lp' == args.spatial_adj_mode:
        temp = train_X
        train_data = temp.permute(0, 2, 1).contiguous().reshape(-1, channel_num)
        Adj_p = torch.tensor(np.corrcoef(train_data.numpy().T, ddof=1), dtype=torch.float32)
        Adj_l = torch.tensor(load_adj('bciciv2a'), dtype=torch.float32)
        Adj = torch.tensor(torch.zeros((22,22)),dtype=torch.float32)
        for i in range(22):
            for j in range(22):
                if Adj_l[i][j] == 0:
                    Adj[i][j] = Adj_p[i][j]
                else:
                    Adj[i][j] = 0.5*Adj_l[i][j]+0.5*Adj_p[i][j]
        
        # Adj = (Adj_p + Adj_l)/2
        print(Adj)
    else:
        raise ValueError('adj_mode only support l,p or random but {} is gotten'.format(args.adj_mode))
    
    model_classifier = STGENET(Adj=Adj, in_chans=channel_num, n_classes=args.n_class, time_window_num=slide_window_num,spatial_GCN=args.spatial_GCN, time_GCN=args.time_GCN , 
                                k_spatial=args.k_spatial, k_time=args.k_time, dropout=args.dropout, input_time_length=slide_window_length,
                                out_chans=args.out_chans,kernel_size=args.kernel_size,slide_window=slide_window_num,sampling_rate=args.sampling_rate,device=args.device)

    print(model_classifier)
    
    print("target_id:{} spatial_GCN:{}  time_GCN:{}".format(args.id, args.spatial_GCN, args.time_GCN))

    optimizer = torch.optim.AdamW(model_classifier.parameters(), lr=args.lr, weight_decay=args.w_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0

    # -------------------------------------------------run------------------------------------------------------------
    acc_list = []
    start_time = time.time()

    train_loader = DataLoader(EEGDataSet(slide_train_X, slide_train_y), batch_size=args.batch_size,shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(EEGDataSet(slide_test_X, slide_test_y), batch_size=args.batch_size,shuffle=True, num_workers=0, drop_last=True)

    transform = build_tranforms()
    rta = RepeatedTrialAugmentation(transform, m=5)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(opt_classifier,step_size=100,gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=2**-12)

    for epoch in range(start_epoch, args.epochs):
        node_weights,space_node_weights = train_one_epoch(epoch, train_loader, (slide_train_X, slide_train_y), model_classifier, args.device, optimizer,
                                   criterion, tensorboard, start_time, args,rta)
        avg_acc, avg_loss = evaluate_one_epoch(epoch, test_loader, (slide_test_X, slide_test_y), model_classifier, args.device,
                                                          criterion, tensorboard, args, start_time,rta)

        acc_list.append(avg_acc)
        save_checkpoints = {'model': model_classifier.state_dict(),
                            'epoch': epoch + 1,
                            'acc': avg_acc}
        if avg_acc > best_acc:
            best_acc = avg_acc
            save(save_checkpoints, os.path.join(args.model_path, 'model_best.pth.tar'))
        print('best_acc:{}'.format(best_acc))
        save(save_checkpoints, os.path.join(args.model_path, 'model_newest.pth.tar'))
        # scheduler.step()

    with open(args.spatial_adj_path+'/spatial_node_weights.txt','a') as f:
        tem = str(space_node_weights)
        f.write(tem)
        f.write('\r\n')
    with open(args.time_adj_path+'/time_node_weights.txt','a')as f:
        tem = str(node_weights)
        f.write(tem)
        f.write('\r\n')
    

    plt.figure()
    plt.plot(acc_list,label='test_acc')
    plt.legend()
    plt.title('Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig(args.result_path+f'/test_acc_{str(args.id)}.png')
    df = pd.DataFrame(acc_list)
    df.to_csv(args.result_path+f'/test_acc_{str(args.id)}.csv',header=0,index=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device',type=int,default=3,help='GPU device.')
    parser.add_argument('-channel_num',type=int,default=22,help='Channel num.')
    parser.add_argument('-n_class',type=int,default=4,help='Class num.')
    parser.add_argument('-data_path',type=str,default='',help='The data path file.')
    parser.add_argument('-id',type=int,default=1,help='Subject id used to train and test.')
    parser.add_argument('-data_type',type=str,default='bci2a',help='Select which data you want to use.')

    parser.add_argument('-out_chans',type=int,default=64,help='Out channels.')
    parser.add_argument('-kernel_size',type=int,default=63,help='Kernel size.')

    parser.add_argument('-spatial_adj_mode',type=str,default='l',choices=['l','p','r'],
                        help='l is defined that adj is initialized based on spatial position of EEG electrodes.'
                             'p is defined that adj is initialized based on Pearson correlation matrix.'
                             'r is defined that adj is initialized based on random matrix.')
    parser.add_argument('-rta',type=bool,default=True,help='Whether to use data argument.')
    
    parser.add_argument('-window_length',type=int,default=125,help='The sliding window length.')
    parser.add_argument('-window_padding',type=int,default=100,help='The padding of sliding window.')
    parser.add_argument('-sampling_rate',type=int,default=250,help='Data sampling rate.')

    parser.add_argument('-spatial_GCN',type=bool,default=True,help='Whether spatial_GCN is selected.')
    parser.add_argument('-time_GCN',type=bool,default=True,help='Whether time_GCN is selected.')

    parser.add_argument('-k_spatial',type=int,default=2,help='The layer of spatial_GCN embedding')
    parser.add_argument('-k_time',type=int,default=2,help='The layer of time_GCN embedding')

    parser.add_argument('-dropout', type=float, default=0.5, help='Dropout rate.')

    parser.add_argument('-epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('-batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('-lr', type=float, default=2 ** -12, help='Learning rate.')
    parser.add_argument('-w_decay', type=float, default=0.01, help='Weight decay.')

    parser.add_argument('-log_path', type=str, default=None, help='The log files path.')
    parser.add_argument('-model_path', type=str, default=None, help='Path of saved model.')
    parser.add_argument('-result_path', type=str, default=None, help='Path of result.')
    parser.add_argument('-spatial_adj_path', type=str, default=None, help='Path of saved spatial_adj.')
    parser.add_argument('-time_adj_path', type=str, default=None, help='Path of saved time_adj.')
    parser.add_argument('-print_freq', type=int, default=1, help='The frequency to show training information.')
    parser.add_argument('-seed', type=int, default='2024', help='Random seed.')

    parser.add_argument('-father_path',type=str,default='')
    
    parser.add_argument('-tensorboard_path', type=str, default=None, help='Path of tensorboard.')

    args_ = parser.parse_args()
    acc_list = []
    for i in range(1,10):
        # args_.data_type = 'bci2b'
        args_.data_type = 'labdata'
        args_.id = i
        start_run(args_)
