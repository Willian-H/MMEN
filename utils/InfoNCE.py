import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
    温度：在计算交叉熵之前，将 Logits 除以温度。
         减少：应用于输出的减少方法。
             值必须是 ['none', 'sum', 'mean'] 之一。
             有关每个选项的更多详细信息，请参阅 torch.nn.function.cross_entropy。
         Negative_mode：确定如何处理（可选）Negative_keys。
             值必须是 ['paired', 'unpaired'] 之一。
             如果“配对”，则每个查询样本都与多个负键配对。
             与三重态损失相当，但每个样本有多个负数。
             如果“未配对”，则负密钥集均与任何正密钥无关。
     输入形状：
         查询：（N，D）带有查询样本的张量（例如输入的嵌入）。
         Positive_key：（N，D）具有正样本的张量（例如增强输入的嵌入）。
         negative_keys（可选）：具有负样本的张量（例如其他输入的嵌入）
             如果 negative_mode = 'paired'，则 negative_keys 是一个 (N, M, D) 张量。
             如果 negative_mode = 'unpaired'，则 negative_keys 是一个 (M, D) 张量。
             如果无，则样本的负键是其他样本的正键。
     返回：
          InfoNCE 损失的价值。
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, hp, seq_len_query, seq_len_positive, temperature=0.1, reduction='mean', negative_mode='unpaired',
                 device=torch.device('cuda:0')):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.device = device
        self.orig_d_l = seq_len_query
        self.orig_d_av = seq_len_positive
        self.d_l, self.d_av = 32, 32 ### 50, 50, 30, 30
        # self.d_l, self.d_av = 50, 50
        # 取query和positive的序列长度的最小值
        # self.d_l, self.d_av = min(seq_len_query, seq_len_positive), min(seq_len_query, seq_len_positive)
        # self.embed_dropout = hp.embed_dropout
        self.embed_dropout = hp.embed_dropout if hp.embed_dropout is not None else 1e-4

        self.info_proj_query = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False, device=device)
        self.info_proj_positive = nn.Conv1d(self.orig_d_av, self.d_av, kernel_size=1, padding=0, bias=False,
                                            device=device)

    def forward(self, query, positive_key, negative_keys=None):
        # x_l_ = F.dropout(query.transpose(1, 2), p=self.embed_dropout, training=self.training)
        # x_av_ = positive_key.transpose(1, 2)
        # 统一为float32
        x_l_ = F.dropout(query, p=self.embed_dropout, training=self.training).to(torch.float32).to(self.device)
        x_av_ = positive_key.to(torch.float32).to(self.device)
        if negative_keys is not None:
            x_m_ = negative_keys.to(torch.float32).to(self.device)
            proj_x_m = x_m_ if self.orig_d_av == self.d_av else self.info_proj_positive(x_m_)
            negative_keys = torch.mean(proj_x_m, dim=-1)

        # Project the textual/visual/audio features
        proj_x_l = x_l_ if self.orig_d_l == self.d_l else self.info_proj_query(x_l_)
        proj_x_av = x_av_ if self.orig_d_av == self.d_av else self.info_proj_positive(x_av_)

        ###消除序列长度的影响,做成二维的方式
        proj_query = torch.mean(proj_x_l, dim=-1)
        proj_positive = torch.mean(proj_x_av, dim=-1)

        return info_nce(proj_query, proj_positive, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    # query dim != positive_key dim
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=torch.device('cuda:0')):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.args = args
        self.device = device

    def forward(self, features, labels=None, mask=None, stop_grad=False, stop_grad_sd=-1.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if stop_grad:
            anchor_dot_contrast_stpg = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T.detach()),
                self.temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # For hard negatives, code adapted from HCL (https://github.com/joshr17/HCL)
        # =============== hard neg params =================
        # tau_plus默认为0.1，beta默认为1.0
        tau_plus = self.args.tau_plus if self.args.tau_plus is not None else 0.1
        beta = self.args.beta if self.args.beta is not None else 1.0
        temperature = 0.5
        N = (batch_size - 1) * contrast_count
        # =============== reweight neg =================
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        exp_logits_neg = exp_logits * (1 - mask) * logits_mask
        exp_logits_pos = exp_logits * mask
        pos = exp_logits_pos.sum(dim=1) / mask.sum(1)

        imp = (beta * (exp_logits_neg + 1e-9).log()).exp()
        reweight_logits_neg = (imp * exp_logits_neg) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_logits_neg.sum(dim=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        log_prob = -torch.log(exp_logits / (pos + Ng))
        # ===============================================

        loss_square = mask * log_prob  # only positive positions have elements

        # mix_square = exp_logits
        mix_square = loss_square

        if stop_grad:
            logits_max_stpg, _ = torch.max(anchor_dot_contrast_stpg, dim=1, keepdim=True)
            logits_stpg = anchor_dot_contrast_stpg - logits_max_stpg.detach()
            # =============== reweight neg =================
            exp_logits_stpg = torch.exp(logits_stpg)
            exp_logits_neg_stpg = exp_logits_stpg * (1 - mask) * logits_mask
            exp_logits_pos_stpg = exp_logits_stpg * mask
            pos_stpg = exp_logits_pos_stpg.sum(dim=1) / mask.sum(1)

            imp_stpg = (beta * (exp_logits_neg_stpg + 1e-9).log()).exp()
            reweight_logits_neg_stpg = (imp_stpg * exp_logits_neg_stpg) / imp_stpg.mean(dim=-1)
            Ng_stpg = ((-tau_plus * N * pos_stpg + reweight_logits_neg_stpg.sum(dim=-1)) / (1 - tau_plus))

            # constrain (optional)
            Ng_stpg = torch.clamp(Ng_stpg, min=N * np.e ** (-1 / temperature))
            log_prob_stpg = -torch.log(exp_logits_stpg / (pos_stpg + Ng_stpg))
            # ===============================================
            tmp_square = mask * log_prob_stpg
        else:
            # tmp_square = exp_logits
            tmp_square = loss_square
        if stop_grad:
            ac_square = stop_grad_sd * tmp_square[batch_size:, 0:batch_size].T + (1 - stop_grad_sd) * tmp_square[
                                                                                                      0:batch_size,
                                                                                                      batch_size:]
        else:
            ac_square = tmp_square[0:batch_size, batch_size:]

        adv_weight = self.args.adv_weight if self.args.adv_weight is not None else 1.0
        mix_square[0:batch_size, batch_size:] = ac_square * adv_weight
        mix_square[batch_size:, 0:batch_size] = ac_square.T * adv_weight

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = mix_square.sum(1) / mask.sum(1)

        # loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ori_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=torch.device('cuda:0')):
        super(ori_SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.args = args
        self.device = device
        self.info_proj = nn.Conv1d(50, 2, kernel_size=1, padding=0, bias=False, device=device)

    def forward(self, features, labels=None, mask=None, stop_grad=False, stop_grad_sd=-1.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # 将数据转为float32
        features = features.to(torch.float32).to(self.device)
        labels = labels.to(torch.float32).to(self.device)
        # n_views指的是两个视角，lables由[-1,1]中隔0.2取值，所以有11个值
        # 将features的第二个维度通过一个卷积层，将维度降低到11
        features = self.info_proj(features)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if stop_grad:
            anchor_dot_contrast_stpg = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T.detach()),
                self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        loss_square = mask * log_prob  # only positive position have elements
        mix_square = loss_square
        if stop_grad:
            logits_max_stpg, _ = torch.max(anchor_dot_contrast_stpg, dim=1, keepdim=True)
            logits_stpg = anchor_dot_contrast_stpg - logits_max_stpg.detach()
            # compute log_prob
            exp_logits_stpg = torch.exp(logits_stpg) * logits_mask
            log_prob_stpg = logits_stpg - torch.log(exp_logits_stpg.sum(1, keepdim=True))
            loss_square_stpg = mask * log_prob_stpg
            tmp_square = loss_square_stpg
        else:
            tmp_square = loss_square
        if stop_grad:
            ac_square = stop_grad_sd * tmp_square[batch_size:, 0:batch_size].T + (1 - stop_grad_sd) * tmp_square[
                                                                                                      0:batch_size,
                                                                                                      batch_size:]
        else:
            ac_square = tmp_square[0:batch_size, batch_size:]

        adv_weight = self.args.adv_weight if self.args.adv_weight is not None else 1.0
        mix_square[0:batch_size, batch_size:] = ac_square * adv_weight
        mix_square[batch_size:, 0:batch_size] = ac_square.T * adv_weight

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = mix_square.sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class SupConLoss1(nn.Module):
    """https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-30)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss