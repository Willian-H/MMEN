import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.abstract_weighting import AbsWeighting


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
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
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, device=torch.device('cuda:0')):
        super(AutomaticWeightedLoss, self).__init__()
        # 权重参数初始化方式：1.全1 2.随机 3.第一个为主任务权重为1，其余多个为辅助任务权重为0.1
        # params = torch.ones((num,), requires_grad=True, device=device)
        # 2.随机,范围为[0,1]
        # params = torch.rand((num,), requires_grad=True, device=device)
        # 3.全部为0.5
        params = torch.ones((num,), requires_grad=True, device=device) * 0.5
        # 第一个为主任务权重为1，其余多个为辅助任务权重为0.1
        # params = torch.tensor([1.0]+ [0.1]*(num-2), requires_grad=True, device=device)
        # params = torch.tensor([1.0, 0.2, 0.8, 0.4], requires_grad=True, device=device)
        # params = torch.tensor([1.0, 0.8, 0.2, 0.6], requires_grad=True, device=device)
        self.params = nn.Parameter(params)
        
    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # Laplace likelihood
            loss_sum += 1 / self.params[i] * loss + torch.log(self.params[i] + 1)
            
            # Gaussion likelihood
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i])
            # loss_sum += 0.5 / (self.params[i]) * loss + torch.log(1 + self.params[i])
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        # +1避免了log 0的问题  log sigma部分对于整体loss的影响不大
        return loss_sum


class AutomaticWeightedLoss2(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2, device=torch.device('cuda:0')):
        super(AutomaticWeightedLoss2, self).__init__()
        # 权重参数初始化方式：1.全1 2.随机 3.第一个为主任务权重为1，其余多个为辅助任务权重为0.1
        # params = torch.ones((num,), requires_grad=True, device=device)
        # 2.随机,范围为[0,1]
        # params = torch.rand((num,), requires_grad=True, device=device)
        # 3.全部为0.5
        params = torch.ones((num,), requires_grad=True, device=device) * 0.5
        # 第一个为主任务权重为1，其余多个为辅助任务权重为0.1
        # params = torch.tensor([1.0]+ [0.1]*(num-2), requires_grad=True, device=device)
        # params = torch.tensor([1.0, 0.2, 0.8, 0.4], requires_grad=True, device=device)
        # params = torch.tensor([1.0, 0.8, 0.2, 0.6], requires_grad=True, device=device)
        self.params = nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # Laplace likelihood
            # loss_sum += 1 / self.params[i] * loss + torch.log(self.params[i] + 1)

            # Gaussion likelihood
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i])
            # loss_sum += 0.5 / (self.params[i]) * loss + torch.log(1 + self.params[i])
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        # +1避免了log 0的问题  log sigma部分对于整体loss的影响不大
        return loss_sum

    
    
class UW(AbsWeighting):
    r"""Uncertainty Weights (UW).

    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_ \
    and implemented by us.
    """
    def __init__(self, task_num, device=torch.device('cuda:0')):
        super(UW, self).__init__()
        self.task_num = task_num
        self.device = device
        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*self.task_num, device=self.device))


    def backward(self, losses, **kwargs):
        # loss = (losses/(2*self.loss_scale.exp())+self.loss_scale/2).sum()
        # return loss
        total_loss = 0
        for i in range(self.task_num):
            total_loss += (losses[i] / (2 * self.loss_scale.exp()[i]) + self.loss_scale[i] / 2)
        return total_loss


class GradNorm(AbsWeighting):
    r"""Gradient Normalization (GradNorm).
    
    This method is proposed in `GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML 2018) <http://proceedings.mlr.press/v80/chen18a/chen18a.pdf>`_ \
    and implemented by us.
    Args:
        alpha (float, default=1.5): The strength of the restoring force which pulls tasks back to a common training rate.
    """
    def __init__(self, task_num, device=torch.device('cuda:0')):
        super(GradNorm, self).__init__()
        self.task_num = task_num
        self.device = device
        
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([1.0]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        alpha = kwargs['alpha']
        if self.epoch >= 1:
            loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)
            grads = self._get_grads(losses, mode='backward')
            if self.rep_grad:
                per_grads, grads = grads[0], grads[1]
                
            G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
            G = G_per_loss.mean(0)
            L_i = torch.Tensor([losses[tn].item()/self.train_loss_buffer[tn, 0] for tn in range(self.task_num)]).to(self.device)
            r_i = L_i/L_i.mean()
            constant_term = (G*(r_i**alpha)).detach()
            L_grad = (G_per_loss-constant_term).abs().sum(0)
            L_grad.backward()
            loss_weight = loss_scale.detach().clone()
            
            if self.rep_grad:
                self._backward_new_grads(loss_weight, per_grads=per_grads)
            else:
                self._backward_new_grads(loss_weight, grads=grads)
            return loss_weight.cpu().numpy()
        else:
            loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
            loss.backward()
            return np.ones(self.task_num)


class GradVac(AbsWeighting):
    r"""Gradient Vaccine (GradVac).

    This method is proposed in `Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models (ICLR 2021 Spotlight) <https://openreview.net/forum?id=F1vEjWK-lH_>`_ \
    and implemented by us.

    Args:
        GradVac_beta (float, default=0.5): The exponential moving average (EMA) decay parameter.
        GradVac_group_type (int, default=0): The parameter granularity (0: whole_model; 1: all_layer; 2: all_matrix).

    .. warning::
            GradVac is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """

    def __init__(self, task_num, device=torch.device('cuda:0')):
        super(GradVac, self).__init__()
        self.task_num = task_num
        self.device = device
        self.step = 0
        self.rep_grad = False

    def init_param(self):
        self.step = 0

    def _init_rho(self, group_type):
        if group_type == 0:  # whole_model
            self.k_idx = [-1]
        elif group_type == 1:  # all_layer
            self.k_idx = []
            for module in self.encoder.modules():
                if len(module._modules.items()) == 0 and len(module._parameters) > 0:
                    self.k_idx.append(sum([w.data.numel() for w in module.parameters()]))
        elif group_type == 2:  # all_matrix
            self._compute_grad_dim()
            self.k_idx = self.grad_index
        else:
            raise ValueError
        self.rho_T = torch.zeros(self.task_num, self.task_num, len(self.k_idx)).to(self.device)

    def backward(self, losses, **kwargs):
        beta = kwargs['GradVac_beta'] if 'GradVac_beta' in kwargs.keys() else 0.5
        group_type = kwargs['GradVac_group_type'] if 'GradVac_group_type' in kwargs.keys() else 0
        if self.step == 0:
            self._init_rho(group_type)

        if self.rep_grad:
            raise ValueError('No support method GradVac with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')  # [task_num, grad_dim]

        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                for k in range(len(self.k_idx)):
                    beg, end = sum(self.k_idx[:k]), sum(self.k_idx[:k + 1])
                    if end == -1:
                        end = grads.size()[-1]
                    rho_ijk = torch.dot(pc_grads[tn_i, beg:end], grads[tn_j, beg:end]) / (
                                pc_grads[tn_i, beg:end].norm() * grads[tn_j, beg:end].norm() + 1e-8)
                    if rho_ijk < self.rho_T[tn_i, tn_j, k]:
                        w = pc_grads[tn_i, beg:end].norm() * (
                                    self.rho_T[tn_i, tn_j, k] * (1 - rho_ijk ** 2).sqrt() - rho_ijk * (
                                        1 - self.rho_T[tn_i, tn_j, k] ** 2).sqrt()) / (grads[tn_j, beg:end].norm() * (
                                    1 - self.rho_T[tn_i, tn_j, k] ** 2).sqrt() + 1e-8)
                        pc_grads[tn_i, beg:end] += grads[tn_j, beg:end] * w
                        # batch_weight[tn_j] += w.item()
                        self.rho_T[tn_i, tn_j, k] = (1 - beta) * self.rho_T[tn_i, tn_j, k] + beta * rho_ijk
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        self.step += 1
        return batch_weight


class Aligned_MTL(AbsWeighting):
    r"""Aligned-MTL.

    This method is proposed in `Independent Component Alignment for Multi-Task Learning (CVPR 2023) <https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/SamsungLabs/MTL>`_.

    """

    def __init__(self, task_num, device=torch.device('cuda:0'), rep_grad=False):
        super(Aligned_MTL, self).__init__()
        self.task_num = task_num
        self.device = device
        self.rep_grad = rep_grad
        # self.rep = None

    def get_share_params(self):
        # Replace this with your actual shared parameters
        shared_params = [param for param in self.parameters()]
        return shared_params

    def zero_grad_share_params(self):
        for param in self.get_share_params():
            if param.grad is not None:
                param.grad.data.zero_()

    def backward(self, losses, **kwargs):
        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]

        M = torch.matmul(grads, grads.t()).to(torch.float32).to(self.device)  # [num_tasks, num_tasks]
        lmbda, V = torch.symeig(M, eigenvectors=True)
        tol = (
                torch.max(lmbda)
                * max(M.shape[-2:])
                * torch.finfo().eps
        )
        rank = sum(lmbda > tol)

        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]

        sigma = torch.diag(1 / lmbda.sqrt())

        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())
        alpha = B.sum(0)

        if self.rep_grad:
            self._backward_new_grads(alpha, per_grads=per_grads)
        else:
            self._backward_new_grads(alpha, grads=grads)
        return alpha.detach().cpu().numpy()


# class IMTL(nn.Module):
#     def __init__(self, method='hybrid'):
#         super().__init__()
#         self.method = method
#         self.s_t = False
#         self.num_losses = -1
#         self.ind = []
#         self.register_buffer('e', torch.exp(torch.ones([1])))
#
#     def instantiate(self,
#                     device,
#                     losses: list[torch.Tensor, ...]):
#         del self.s_t
#         self.device = data.device
#         for i, loss in enumerate(losses):
#             self.ind.append(-1)
#             if loss.requires_grad:
#                 self.num_losses += 1
#                 self.ind[-1] = self.num_losses
#
#         self.register_parameter('s_t', nn.Parameter(torch.ones(self.num_losses + 1).squeeze(), requires_grad=True))
#
#     def forward(self,
#                 data: torch.Tensor,
#                 losses: list[torch.Tensor, ...],
#                 ) -> tuple([torch.Tensor, ...]):
#         if isinstance(self.s_t, bool): self.instantiate(data.device, losses)
#
#         # >>> Loss Balance
#         L_t = torch.empty(self.s_t.numel(), device=self.device)
#         g_t = torch.empty([self.num_losses + 1, 1, data.shape[-1]], device=self.device)
#         for i, loss in zip(self.ind, losses):
#             if loss.requires_grad:
#                 L_t[i] = loss * self.e.pow(self.s_t[i]) - self.s_t[i]
#                 g_t[i, ::] = torch.autograd.grad(L_t[i], data, retain_graph=True, create_graph=True)[0].mean(dim=0,
#                                                                                                              keepdim=True)
#         u_t = g_t / (torch.linalg.norm(g_t, 2, (-1, -2)) + 1e-6).unsqueeze(-1).unsqueeze(-1)
#
#         # >>> Gradient Balance
#         D = g_t[0, ::].unsqueeze(0).repeat(self.num_losses, 1, 1) - g_t[1:, ::]
#         UT = u_t[0, :].unsqueeze(0).repeat(self.num_losses, 1, 1).mT - u_t[1:, ::].mT
#
#         alpha_2T = g_t[0, ::].unsqueeze(0).matmul(UT).matmul(torch.linalg.pinv(D.matmul(UT)))
#         alpha = torch.cat([torch.ones([1, alpha_2T.shape[1], alpha_2T.shape[2]], device=self.device) -
#                            alpha_2T.sum(dim=0, keepdim=True), alpha_2T], dim=0).squeeze()
#
#         if self.method == 'hybrid':
#             return torch.sum(L_t * alpha), [loss.backward(retain_graph=True) for i, loss in enumerate(L_t)]
#         elif self.method == 'gradient':
#             return torch.sum(L_t * alpha)
#         elif self.method == 'loss':
#             return [loss.backward(retain_graph=True) for i, loss in enumerate(L_t)]