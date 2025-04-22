import os
import gc
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from data.load_data import MMDataset
from config.config_tune import ConfigTune
from config.config_regression import ConfigRegression


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def run(args):
    config = ConfigRegression(args)
    args = config.get_config()
    
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.seed = 1111
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.tasks}-{args.datasetName}-{args.seed}.pth')
    
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # add tmp tensor to increase the temporary consumption of GPU
    # tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load data and models
 #    evaluate_lst = ['video_0028$_$0022', 'video_0025$_$0023','video_0053$_$0038','video_0054$_$0109',
 #                   'test15$_$00013','test8$_$00006','test5$_$00013','test5$_$00001','video_0054$_$0040']
 #    lst = ['video_0001$_$0031',
 # 'video_0005$_$0002',
 # 'video_0013$_$0035',
 # 'video_0020$_$0005',
 # 'video_0024$_$0062',
 # 'video_0035$_$0011',
 # 'video_0048$_$0022',
 # 'video_0053$_$0004',
 # 'video_0053$_$0050',
 # 'video_0054$_$0032',
 # 'video_0054$_$0040',
 # 'video_0054$_$0041',
 # 'video_0054$_$0052',
 # 'video_0054$_$0116',
 # 'video_0055$_$0005',
 # 'video_0056$_$0010',
 # 'aqgy3_0005$_$00042',
 # 'aqgy3_0007$_$00011',
 # 'aqgy4_0004$_$00009',
 # 'aqgy5_0020$_$00006',
 # 'aqgy5_0025$_$00004',
 # 'aqgy5_0027$_$00014',
 # 'aqgy5_0036$_$00007',
 # 'test1$_$00008',
 # 'test2$_$00021',
 # 'test4$_$00001',
 # 'test5$_$00001',
 # 'test5$_$00007',
 # 'test5$_$00013',
 # 'test5$_$00019',
 # 'test8$_$00003',
 # 'test8$_$00005',
 # 'test8$_$00006',
 # 'test10$_$00002',
 # 'test11$_$00004',
 # 'test11$_$00015',
 # 'test12$_$00000',
 # 'test12$_$00005',
 # 'test12$_$00020',
 # 'test12$_$00021',
 # 'test12$_$00028',
 # 'test12$_$00029',
 # 'test13$_$00005',
 # 'test13$_$00008',
 # 'test14$_$00005',
 # 'test14$_$00006',
 # 'test15$_$00013']
    dataset = MMDataset(args, mode='train')
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)
    model.load_state_dict(torch.load(args.model_save_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_data in dataset:
            vision = batch_data['vision'].to(args.device)
            audio = batch_data['audio'].to(args.device)
            text = batch_data['text'].to(args.device)
            labels = batch_data['labels']
            for k in labels.keys():
                if args.train_mode == 'classification':
                    labels[k] = labels[k].to(args.device).view(-1).long()
                else:
                    labels[k] = labels[k].to(args.device).view(-1, 1)
            outputs = model(text.unsqueeze(0), audio.unsqueeze(0), vision.unsqueeze(0))

            dis = 0.0
            for m in 'MTAV':
                dis += abs(outputs[m]-labels[m])
            if dis < 0.1:
                print(batch_data['id'])
                print(outputs)
                print(batch_data['labels'])

                
    dataset = MMDataset(args, mode='valid')
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)
    model.load_state_dict(torch.load(args.model_save_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_data in dataset:
            vision = batch_data['vision'].to(args.device)
            audio = batch_data['audio'].to(args.device)
            text = batch_data['text'].to(args.device)
            labels = batch_data['labels']
            for k in labels.keys():
                if args.train_mode == 'classification':
                    labels[k] = labels[k].to(args.device).view(-1).long()
                else:
                    labels[k] = labels[k].to(args.device).view(-1, 1)
            outputs = model(text.unsqueeze(0), audio.unsqueeze(0), vision.unsqueeze(0))

            dis = 0.0
            for m in 'MTAV':
                dis += abs(outputs[m]-labels[m])
            if dis < 0.1:
                print(batch_data['id'])
                print(outputs)
                print(batch_data['labels'])
                
                
    dataset = MMDataset(args, mode='test')
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)
    model.load_state_dict(torch.load(args.model_save_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_data in dataset:
            vision = batch_data['vision'].to(args.device)
            audio = batch_data['audio'].to(args.device)
            text = batch_data['text'].to(args.device)
            labels = batch_data['labels']
            for k in labels.keys():
                if args.train_mode == 'classification':
                    labels[k] = labels[k].to(args.device).view(-1).long()
                else:
                    labels[k] = labels[k].to(args.device).view(-1, 1)
            outputs = model(text.unsqueeze(0), audio.unsqueeze(0), vision.unsqueeze(0))

            dis = 0.0
            for m in 'MTAV':
                dis += abs(outputs[m]-labels[m])
            if dis < 0.1:
                print(batch_data['id'])
                print(outputs)
                print(batch_data['labels'])
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=True,
                        help='tune parameters ?')
    parser.add_argument('--modelName', type=str, default='v6',
                        help='support v1/v2/v3/v1_semi')
    parser.add_argument('--datasetName', type=str, default='sims3l',
                        help='support sims3/sims3l')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/baseline',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--supvised_nums', type=int, default=2722,
                        help='number of supervised data')
    parser.add_argument('--tasks', type=str, default='MTAV',
                        help='training tasks')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
#     args.tasks = "M"
#     args.seeds = [1111,1112,1113,1114,1115]
    args.seeds = 1111
    print("modelName:", args.modelName)
    run(args)

