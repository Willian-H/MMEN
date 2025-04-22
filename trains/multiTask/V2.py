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
from utils.losses import *
from utils.gradvac_amp import GradVacAMP
from utils.InfoNCE import *
from utils.adatask import Adam_with_AdaTask
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

def mixGen3(audio_x, video_x, text_x, y_a, y_v, y_t, y_m, alpha=1.0):
    '''
    Args:
        audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
        video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
        text_x: tensor of shape (batch_size, sequence_len, text_in)
        beta: mix_batch_size / batch_size
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    batch_size = audio_x.size()[0]
    index = (torch.arange(batch_size) + 1) % batch_size
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

def mixGen2(audio_x, video_x, text_x, y_a, y_v, y_t, y_m, alpha=1.0, beta=0.5):
    '''
    Args:
        audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
        video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
        text_x: tensor of shape (batch_size, sequence_len, text_in)
        beta: mix_batch_size / batch_size
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    batch_size = audio_x.size()[0]
    mix_batch_size = int(batch_size * beta)

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
        'M': y_m_mix[:mix_batch_size],
        'T': y_t_mix[:mix_batch_size],
        'A': y_a_mix[:mix_batch_size],
        'V': y_v_mix[:mix_batch_size]
    }
    return mixed_a[:mix_batch_size], mixed_v[:mix_batch_size], mixed_t[:mix_batch_size], res

def mixGen(audio_x, video_x, text_x, y_a, y_v, y_t, y_m, alpha=0.1):
    '''
    Args:
        audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
        video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
        text_x: tensor of shape (batch_size, sequence_len, text_in)
        应用于嵌入级别 embedding-level MixGen
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    # lam = 0.5
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


def mixGen4(audio_x, video_x, text_x, y_a, y_v, y_t, y_m, alpha=0.1):
    '''
    Args:
        audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
        video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
        text_x: tensor of shape (batch_size, sequence_len, text_in)
        应用于嵌入级别 embedding-level MixGen
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5
    # lam = 0.5
    batch_size = audio_x.size()[0]
    neg_index = torch.where(y_m < -0.3)
    neu_index = torch.where((y_m >= -0.3) & (y_m <= 0.3))
    pos_index = torch.where(y_m > 0.3)
    neg_index_rand = neg_index[0][torch.randperm(neg_index[0].size()[0])]
    neu_index_rand = neu_index[0][torch.randperm(neu_index[0].size()[0])]
    pos_index_rand = pos_index[0][torch.randperm(pos_index[0].size()[0])]
    # 根据打乱后的索引根据原本的索引进行拼接
    index = torch.zeros(batch_size, dtype=torch.long)
    for i in range(neg_index[0].size()[0]):
        index[neg_index[0][i]] = neg_index_rand[i]
    for i in range(neu_index[0].size()[0]):
        index[neu_index[0][i]] = neu_index_rand[i]
    for i in range(pos_index[0].size()[0]):
        index[pos_index[0][i]] = pos_index_rand[i]
    # 随机打乱
    # index = torch.randperm(batch_size)
    x = audio_x[index, :]
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
    x1, y1, z1 = neg_index[0].size()[0], neu_index[0].size()[0], pos_index[0].size()[0]
    neg_index_mix = torch.where(y_m_mix < -0.3)
    neu_index_mix = torch.where((y_m_mix >= -0.3) & (y_m_mix <= 0.3))
    pos_index_mix = torch.where(y_m_mix > 0.3)
    x2, y2, z2 = neg_index_mix[0].size()[0], neu_index_mix[0].size()[0], pos_index_mix[0].size()[0]
    res = {
        'M': y_m_mix,
        'T': y_t_mix,
        'A': y_a_mix,
        'V': y_v_mix
    }
    return mixed_a, mixed_v, mixed_t, res

# using multi task learning methods
# UW GradNorm
class V2():
    def __init__(self, args):
        assert args.datasetName == 'sims3' or args.datasetName == 'sims3l'

        self.args = args
        # MTAV task
        self.args.tasks = "MAVT"
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        # if isinstance(model, torch.nn.DataParallel):
        #     model = model.module

        # 清理缓存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        # bert_params = [p for n, p in bert_params]
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        # auto weight loss
        awl = AutomaticWeightedLoss(num=len(self.args.tasks), device=self.args.device)
        awl_MICL = AutomaticWeightedLoss2(num=3, device=self.args.device)
        # awl = GradVac(task_num=len(self.args.tasks), device=self.args.device)
        model.Model.set_awl(awl)
        # 如果set_awl_MICL存在，则设置awl_MICL
        if hasattr(model.Model, 'set_awl_MICL'):
            model.Model.set_awl_MICL(awl_MICL)

        # InfoNCE loss for self-supervised learning
        info_nce_tv = InfoNCE(self.args,50,232, device=self.args.device)
        info_nce_ta = InfoNCE(self.args,50,925, device=self.args.device)
        info_nca_mt = InfoNCE(self.args,1207,50, device=self.args.device)
        info_nca_mv = InfoNCE(self.args,1207,232, device=self.args.device)
        info_nca_ma = InfoNCE(self.args,1207,925, device=self.args.device)
        info_nce_2 = InfoNCE(self.args,1207,1207, device=self.args.device)

        # SupConLoss for Supervised Contrastive Learning
        sup_con_loss = SupConLoss(self.args, device=self.args.device)
        ori_sup_con_loss = ori_SupConLoss(self.args, device=self.args.device)

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other},
            {'params': awl.parameters(), 'weight_decay': 0},
            {'params': awl_MICL.parameters(), 'weight_decay': 0}
        ]
        # optimizer_grouped_parameters = [
        #     {'params': bert_params, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
        #     {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
        #     {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
        #     {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other},
        #     {'params': awl.parameters(), 'weight_decay': 0},
        # ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        # optimizer = Adam_with_AdaTask([dict(params=model.parameters(), lr=lr)], n_tasks=2, args=args, device=device)
        # optimizer = Adam_with_AdaTask(optimizer_grouped_parameters, n_tasks=len(self.args.tasks), args=self.args, device=self.args.device)

        # scaler = torch.cuda.amp.GradScaler()
        # grad_optimizer = GradVacAMP(num_tasks=len(self.args.tasks), optimizer=optimizer, DEVICE=self.args.device, beta = 1e-2, reduction='sum', cpu_offload=False)
        # print('using grad_optimizer:', grad_optimizer, flush=True)

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
                    with autocast():    # 混合精度训练
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
                            lst.append(self.criterion(outputs[m], labels[m]))  # lst存的是每个任务的loss
                        loss = model.Model.awl(lst)  # 计算uw的loss，这里计算原始数据的loss
                        # loss = sum(lst)
                        # loss = model.Model.awl.backward(lst)
                        # grad_optimizer.backward(lst)
                        # grad_optimizer.step()
                        # lst1 = lst
                        # loss= sum(lst)

                        # 数据增强部分，alpha=2时效果最好
                        audio_mix, video_mix, text_mix, mix_labels = mixGen(audio, vision, text, labels['A'], labels['V'], labels['T'], labels['M'], alpha=1)
                        # mix outputs
                        outputs_mix = model(text_mix, audio_mix, video_mix)
                        lst = []
                        for m in self.args.tasks:
                            lst.append(self.criterion(outputs_mix[m], mix_labels[m]))  # lst存的是每个任务的loss
                        # weighted sum loss
                        loss_mix = model.Model.awl(lst)     # 计算uw的loss，这里计算增强数据部分的loss
                        # loss_mix = sum(lst)
                        # loss_mix = model.Model.awl.backward(lst)
                        # 将lst1和lst2的loss加起来
                        # grad_optimizer.backward(lst)
                        # lst2 = lst
                        # for(i, j) in zip(lst1, lst2):
                        #     lst.append(i+j)
                        # loss_mix = sum(lst2)
                        # compute loss
                        loss += loss_mix

                    # Inter-modality Contrastive Learning
                    if self.args.use_MICL:
                        # 原始数据部分
                        # tv_loss = info_nce_tv(outputs['Feature_t'], outputs['Feature_v'])
                        # ta_loss = info_nce_ta(outputs['Feature_t'], outputs['Feature_a'])
                        # # loss += 0.5 * tv_loss + 0.5 * ta_loss
                        # loss += self.args.ICL_alpha * tv_loss + (1 - self.args.ICL_alpha) * ta_loss
                        mt_loss = info_nca_mt(outputs['Fusion'], outputs['Feature_t'])
                        ma_loss = info_nca_ma(outputs['Fusion'], outputs['Feature_a'])
                        mv_loss = info_nca_mv(outputs['Fusion'], outputs['Feature_v'])
                        # loss += self.args.MICL_alpha[0] * mt_loss + self.args.MICL_alpha[1] * ma_loss + self.args.MICL_alpha[2] * mv_loss
                        loss += model.Model.awl_MICL([mt_loss, ma_loss, mv_loss])
                        # loss += mt_loss + ma_loss + mv_loss


                        # 增强数据部分
                        # tv_loss_mix = info_nce_tv(outputs_mix['Feature_t'], outputs_mix['Feature_v'])
                        # ta_loss_mix = info_nce_ta(outputs_mix['Feature_t'], outputs_mix['Feature_a'])
                        # # loss += 0.5 * tv_loss_mix + 0.5 * ta_loss_mix
                        # loss += self.args.ICL_alpha * tv_loss_mix + (1 - self.args.ICL_alpha) * ta_loss_mix
                        mt_loss_mix = info_nca_mt(outputs_mix['Fusion'], outputs_mix['Feature_t'])
                        ma_loss_mix = info_nca_ma(outputs_mix['Fusion'], outputs_mix['Feature_a'])
                        mv_loss_mix = info_nca_mv(outputs_mix['Fusion'], outputs_mix['Feature_v'])
                        # loss += self.args.MICL_alpha[0] * mt_loss_mix + self.args.MICL_alpha[1] * ma_loss_mix + self.args.MICL_alpha[2] * mv_loss_mix
                        loss += model.Model.awl_MICL([mt_loss_mix, ma_loss_mix, mv_loss_mix])
                        # loss += mt_loss_mix + ma_loss_mix + mv_loss_mix

                        # 原始数据部分
                        # tv_loss = sup_con_loss(outputs['Feature_t'], labels['T'])
                        # labels 的大小应该是 (bsz,)
                        # tv_loss = ori_sup_con_loss(outputs['Feature_t'], labels['T'].view(-1))
                        # loss += tv_loss

                        # mt_token_loss = info_nce(outputs['Fusion_token'].to(torch.float32), outputs['Feature_t_token'].to(torch.float32))
                        # ma_token_loss = info_nce(outputs['Fusion_token'].to(torch.float32), outputs['Feature_a_token'].to(torch.float32))
                        # mv_token_loss = info_nce(outputs['Fusion_token'].to(torch.float32), outputs['Feature_v_token'].to(torch.float32))
                        # loss += self.args.MICL_alpha[0] * mt_token_loss + self.args.MICL_alpha[1] * ma_token_loss + self.args.MICL_alpha[2] * mv_token_loss

                    # if self.args.use_IBCL:
                    #     # 将labels['annotations']根据三类情感（-1, 0, -1）分别分成三个tensor
                    #     # neg_index = torch.where(labels['annotations'] == -1)
                    #     # neu_index = torch.where(labels['annotations'] == 0)
                    #     # pos_index = torch.where(labels['annotations'] == 1)
                    #     neg_index = torch.where(labels['M'] < -0.3)
                    #     neu_index = torch.where((labels['M'] >= -0.3) & (labels['M'] <= 0.3))
                    #     pos_index = torch.where(labels['M'] > 0.3)
                    #     # 根据上面的索引将feature在第一个维度上进行切分
                    #     neg_feature_m = outputs['Fusion'][neg_index[0]]
                    #     neu_feature_m = outputs['Fusion'][neu_index[0]]
                    #     pos_feature_m = outputs['Fusion'][pos_index[0]]
                    #     # 将数据增强部分的同一类别的feature拼接起来作为正样本 todo
                    #     # 拼接另外两个类别的feature作为负样本
                    #     neg_neg_feature_m = torch.cat((neu_feature_m, pos_feature_m), 0)
                    #     neu_neg_feature_m = torch.cat((neg_feature_m, pos_feature_m), 0)
                    #     pos_neg_feature_m = torch.cat((neg_feature_m, neu_feature_m), 0)
                    #     # 将单个类别的feature在第一个维度上进行打乱作为正样本
                    #     neg_pos_feature_m = neg_feature_m[torch.randperm(neg_feature_m.size()[0])]
                    #     neu_pos_feature_m = neu_feature_m[torch.randperm(neu_feature_m.size()[0])]
                    #     pos_pos_feature_m = pos_feature_m[torch.randperm(pos_feature_m.size()[0])]
                    #     # 将相同类别的样本模态拉进，不同类别的样本模态拉远
                    #     neg_loss = info_nce_2(neg_feature_m, neg_pos_feature_m, neg_neg_feature_m)
                    #     neu_loss = info_nce_2(neu_feature_m, neu_pos_feature_m, neu_neg_feature_m)
                    #     pos_loss = info_nce_2(pos_feature_m, pos_pos_feature_m, pos_neg_feature_m)
                    #     loss += self.args.IBCL_alpha[0] * neg_loss + self.args.IBCL_alpha[1] * neu_loss + self.args.IBCL_alpha[2] * pos_loss
                    #
                    #     # 增强数据部分
                    #     # 将mix_labels根据值分为三类情感分别分成三个tensor
                    #     # [-1,-0.2],[-0.2,0.2],[0.2,1]进行切分
                    #     neg_index_mix = torch.where(mix_labels['M'] < -0.3)
                    #     neu_index_mix = torch.where((mix_labels['M'] >= -0.3) & (mix_labels['M'] <= 0.3))
                    #     pos_index_mix = torch.where(mix_labels['M'] > 0.3)
                    #     # 根据上面的索引将feature在第一个维度上进行切分
                    #     neg_feature_m_mix = outputs_mix['Fusion'][neg_index_mix[0]]
                    #     neu_feature_m_mix = outputs_mix['Fusion'][neu_index_mix[0]]
                    #     pos_feature_m_mix = outputs_mix['Fusion'][pos_index_mix[0]]
                    #     # 拼接另外两个类别的feature作为负样本
                    #     neg_neg_feature_m_mix = torch.cat((neu_feature_m_mix, pos_feature_m_mix), 0)
                    #     neu_neg_feature_m_mix = torch.cat((neg_feature_m_mix, pos_feature_m_mix), 0)
                    #     pos_neg_feature_m_mix = torch.cat((neg_feature_m_mix, neu_feature_m_mix), 0)
                    #     # 将单个类别的feature在第一个维度上进行打乱作为正样本
                    #     neg_pos_feature_m_mix = neg_feature_m_mix[torch.randperm(neg_feature_m_mix.size()[0])]
                    #     neu_pos_feature_m_mix = neu_feature_m_mix[torch.randperm(neu_feature_m_mix.size()[0])]
                    #     pos_pos_feature_m_mix = pos_feature_m_mix[torch.randperm(pos_feature_m_mix.size()[0])]
                    #     # 将相同类别的样本模态拉进，不同类别的样本模态拉远
                    #     neg_loss_mix = info_nce_2(neg_feature_m_mix, neg_pos_feature_m_mix, neg_neg_feature_m_mix)
                    #     neu_loss_mix = info_nce_2(neu_feature_m_mix, neu_pos_feature_m_mix, neu_neg_feature_m_mix)
                    #     pos_loss_mix = info_nce_2(pos_feature_m_mix, pos_pos_feature_m_mix, pos_neg_feature_m_mix)
                    #     loss += self.args.IBCL_alpha[0] * neg_loss_mix + self.args.IBCL_alpha[1] * neu_loss_mix + self.args.IBCL_alpha[2] * pos_loss_mix
                    #
                    #     # neg_loss = info_nce_2(neg_feature_m, neg_feature_m_mix, neg_neg_feature_m)
                    #     # neu_loss = info_nce_2(neu_feature_m, neu_feature_m_mix, neu_neg_feature_m)
                    #     # pos_loss = info_nce_2(pos_feature_m, pos_feature_m_mix, pos_neg_feature_m)
                    #     # loss += self.args.IBCL_alpha[0] * neg_loss + self.args.IBCL_alpha[1] * neu_loss + self.args.IBCL_alpha[2] * pos_loss


                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # optimizer.backward_and_step(
                    #     losses=losses,
                    #     shared_parameters=list(model.shared_parameters()),
                    #     task_specific_parameters=list(model.task_specific_parameters()),
                    #     last_shared_parameters=list(model.last_shared_parameters()),
                    # )
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

            train_results_lst.append(train_results)
            # add results to list
            valid_results_lst.append(val_results)


            cur_valid = val_results[self.args.tasks[0]][self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            print(list(awl.parameters()), list(awl_MICL.parameters()))

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
                    # loss = model.Model.awl.backward(lst)
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
