import os
import time
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from torch.autograd import Variable
from utils.losses import AutomaticWeightedLoss
logger = logging.getLogger('MSA')

    
def mixup_data_no_grad(x, y, y_m, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam  = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    y_m_a, y_m_b = y_m, y_m[index]
    return mixed_x, y_a, y_b, y_m_a, y_m_b, lam

def mixGen(audio_x, video_x, text_x, y_a, y_v, y_t, y_m, alpha=1.0):
    '''
    Args:
        audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
        video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
        text_x: tensor of shape (batch_size, sequence_len, text_in)
    '''
    if alpha > 0:
        lam  = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    batch_size = audio_x.size()[0]
    index = torch.randperm(batch_size)
    mixed_a = lam * audio_x + (1 - lam) * audio_x[index, :]
    mixed_v = lam * video_x + (1 - lam) * video_x[index, :]
    # concate for text sequence
    mixed_t = torch.cat((text_x, text_x[index, :]), dim=1)
    
    # label mix
    y_a_mix = lam * y_a + (1 - lam) * y_a[index, :] 
    y_v_mix = lam * y_v + (1 - lam) * y_v[index, :] 
    y_m_mix = lam * y_m + (1 - lam) * y_m[index, :] 
    
    y_t_mix = lam * y_t + (1 - lam) * y_t[index, :] 
    # y_t_mix = 0.5 * y_t + 0.5 * y_t[index, :] 

    res = {
        'M': y_m_mix,
        'T': y_t_mix,
        'A': y_a_mix,
        'V': y_v_mix
    }
    return mixed_a, mixed_v, mixed_t, res

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# using multi task learning methods
# UW GradNorm
class V3():
    def __init__(self, args):
        assert args.datasetName == 'sims3' or args.datasetName == 'sims3l'

        self.args = args
        self.args.tasks = "MTAV"
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        main_params = list(model.Model.trans_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        main_params = [p for n, p in main_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'trans_model' not in n]
        
        # auto weight loss
        awl = AutomaticWeightedLoss(num=len(self.args.tasks), device=self.args.device)
        model.Model.set_awl(awl)
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': main_params, 'weight_decay': self.args.weight_decay_main, 'lr': self.args.learning_rate_main},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other},
            {'params': awl.parameters(), 'weight_decay': 0},
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        
        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        train_results_lst = []
        valid_results_lst = []
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)

                    lst = []
                    for m in self.args.tasks:
                        lst.append(self.criterion(outputs[m], labels[m]))
                    loss = model.Model.awl(lst)

                    audio_mix, video_mix, text_mix, mix_labels = mixGen(audio, vision, text, labels['A'], labels['V'], labels['T'], labels['M'])
                    # mix outputs
                    outputs_mix = model(text_mix, audio_mix, video_mix)
                    lst = []
                    for m in self.args.tasks:
                        lst.append(self.criterion(outputs_mix[m], mix_labels[m]))
                    # weighted sum loss  
                    loss_mix = model.Model.awl(lst)              
                    # compute loss
                    loss += loss_mix
                    
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        # y_true[m].append(labels['M'].cpu())
                        y_true[m].append(labels[m].cpu())
            train_loss = train_loss / len(dataloader['train'])

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss))
            
            train_results = {}
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(results))
                train_results[m] = results
            train_results['Loss'] = train_loss

            # validation, using valid set
            val_results = self.do_test(model, dataloader['valid'], mode="TEST")
            # test_results = self.do_test(model, dataloader['test'], mode="TEST")

            # add results to list
            train_results_lst.append(train_results)
            valid_results_lst.append(val_results)
            
            
            cur_valid = val_results[self.args.tasks[0]][self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            print(list(awl.parameters()))
            
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            
            # early stop, at least 20 epochs
            if (epochs - best_epoch >= self.args.early_stop) and (epochs > 20):
                return train_results_lst, valid_results_lst

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    outputs = model(text, audio, vision)
                    loss = 0.0
                    lst = []
                    for m in self.args.tasks:
                        lst.append(self.criterion(outputs[m], labels[m]))
                    loss = model.Model.awl(lst)
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        # y_true[m].append(labels['M'].cpu())
                        y_true[m].append(labels[m].cpu())
        eval_loss = round(eval_loss / len(dataloader), 4)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        eval_results = {}
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            logger.info('%s: >> ' %(m) + dict_to_str(results))
            eval_results[m] = results
        # eval_results = eval_results[self.args.tasks[0]]
        eval_results['Loss'] = eval_loss
        return eval_results
