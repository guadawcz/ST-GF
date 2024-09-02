import torch
import math
import sys
import time
import datetime
from .tools import AverageMeter, accuracy
from thop import profile

def train_one_epoch(epoch, iterator, data, model, device, optimizer, criterion, tensorboard, start_time, args, rta):
    print('--------------------------Start training at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)

    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    criterion = criterion.to(device)

    model.train()
    data, data_labels = data
    steps = data.shape[0] // args.batch_size + 1 if data.shape[0] % args.batch_size else data.shape[0] // args.batch_size
    step = 0
    for features, labels in iterator:
        features,labels = rta(features,labels)
        features = features.to(device)
        labels = labels.to(device)

        predicts,node_weights,space_node_weights = model(features)
        # flops, params = profile(model, (features,))
        # print(flops)
        # print(params)
        loss = criterion(predicts, labels)

        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log['loss'].update(loss.item(), len(features))
        dict_log['acc'].update(acc.item(), len(features))
        optimizer .zero_grad()
        loss.backward()
        optimizer .step()
        all_steps = epoch * steps + step + 1
        if 0 == (all_steps % args.print_freq):
            lr = list(optimizer .param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]
            print_information = 'id:{}   time consumption:{}    epoch:{}/{}  lr:{}    '.format(
                args.id, et, epoch + 1, args.epochs, lr)
            for key, value in dict_log.items():
                loss_info = "{}(val/avg):{:.3f}/{:.3f}  ".format(key, value.val, value.avg)
                print_information = print_information + loss_info
                tensorboard.add_scalar(key, value.val, all_steps)
            print(print_information)
        step = step + 1
    
    print('--------------------------End training at epoch:{}--------------------------'.format(epoch + 1))
    return node_weights,space_node_weights


def evaluate_one_epoch(epoch, iterator, data, model, device, criterion, tensorboard, args, start_time,rta):
    print('--------------------------Start evaluating at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    model.eval()
    data, data_labels = data
    step = 0
    start_time = time.time()
    for features, labels in iterator:
        features,labels = rta(features,labels)
                # x = x.permute(0,2,1,3)
        # x = x.contiguous().view(x.shape[0],x.shape[1],-1)
        # features = features.permute(0,2,1,3)
        # features = features.contiguous().view(features.shape[0],features.shape[1],-1)

        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predicts,_,_ = model(features)
            loss = criterion(predicts, labels)
        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))
    end_time = time.time()
    now_time = time.time() - start_time
    et = str(datetime.timedelta(seconds=now_time))[:-7]
    print_information = 'time consumption:{}    epoch:{}/{}   '.format(et, epoch + 1, args.epochs, len(data))
    
    for key, value in dict_log.items():
        loss_info = "{}(avg):{:.3f} ".format(key, value.avg)
        print_information = print_information + loss_info
        tensorboard.add_scalar(key, value.val, epoch)
    
    duration_time = '    ' + str(end_time - start_time)
    print(print_information+duration_time)
    print('--------------------------Ending evaluating at epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg
