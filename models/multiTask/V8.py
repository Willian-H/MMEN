from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder
from models.subNets.transformers_encoder.my_transformer import Encoder
from models.subNets.transformers_encoder.transformer_vit import VitEncoder
# from models.subNets.transformers_encoder.vivit import ViVitEncoder
from models.subNets.transformers_encoder.DST import DST_Backbone
from models.subNets.transformers_encoder.p2t import *

from models.subNets.FusionNetwork import LMF
from models.subNets.FusionNetwork import Concat
from models.subNets.FusionNetwork import TransFN
import numpy as np

__all__ = ['V8']


class MyAVsubNet(nn.Module):
    def __init__(self, seq_len, in_size, hidden_size, dropout, num_heads=4, layers=6, ensemble_depth=1):
        # seq_len：序列长度。
        # in_size：输入特征的维度。
        # hidden_size：隐藏层特征的维度。
        # dropout：dropout概率，用于正则化。
        # num_heads：注意力头的数量。
        # layers：编码器的层数。
        # ensemble_depth：集成深度，即融合策略的层数。
        super(MyAVsubNet, self).__init__()

        self.backbone = Encoder(d_model=hidden_size, embed_len=seq_len, n_head=num_heads, n_layers=layers,
                                drop_prob=dropout,class_token=True)
        # DST作为backbone
        # self.backbone = DST_Backbone(input_dim=hidden_size, length=seq_len, num_layers=layers, num_heads=num_heads,
        #                              dropout=dropout)
        # self.temporal_fusion = nn.Conv1d(seq_len, 50, kernel_size=1, padding=0, bias=True)
        # define the pre-fusion subnetworks
        # 线性层（全连接层），用于将输入特征投影到与隐藏层特征维度相同的空间
        self.proj = nn.Linear(in_size, hidden_size)
        self.pooling_mode = "mean"
        self.d = ensemble_depth

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        # 这是一个用于特征融合的方法，根据提供的池化模式对隐藏状态进行融合。可以选择平均、求和或最大池化。
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_length, in_size)
        '''
        # x = x.transpose(1, 2)  # B N D_in -> B D_in N
        x = self.proj(x)  # B D_in N - > B D_hidden N
        # x = x.transpose(1, 2)  # B D_hidden N -> B N D_hidden
        x, hidden_states = self.backbone(x)
        # _x除去cls,取后面的高级特征
        _x = x[:, 1:, :]
        if self.d == 1:
            # x = self.merged_strategy(x, self.pooling_mode)
            # 取第一个为cls
            x = x[:, 0, :]
        else:
            # l = len(hidden_states)
            # ensemble_lst = []
            # for i in range(self.d):
            #     hidden_state = self.merged_strategy(hidden_states[l - self.d + i], mode=self.pooling_mode)
            #     ensemble_lst.append(hidden_state)
            # x = torch.stack(ensemble_lst, dim=1)
            x = x.mean(dim=1)
        return x, _x


# class MyAVsubNet_V(nn.Module):
#     def __init__(self, seq_len, in_size, hidden_size, dropout, num_heads=4, layers=6, ensemble_depth=1):
#         # seq_len：序列长度。
#         # in_size：输入特征的维度。
#         # hidden_size：隐藏层特征的维度。
#         # dropout：dropout概率，用于正则化。
#         # num_heads：注意力头的数量。
#         # layers：编码器的层数。
#         # ensemble_depth：集成深度，即融合策略的层数。
#         super(MyAVsubNet_V, self).__init__()
#
#         self.backbone = Encoder(d_model=hidden_size, embed_len=seq_len, n_head=num_heads, n_layers=layers, drop_prob=dropout,class_token=True)
#         # 使用VIT作为backbone
#         # self.backbone = VitEncoder(d_model=hidden_size, embed_len=seq_len, n_head=num_heads, n_layers=layers, drop_prob=dropout)
#         # DST作为backbone
#         # self.backbone = DST_Backbone(input_dim=hidden_size, length=seq_len, num_layers=layers, num_heads=num_heads,
#         #                              dropout=dropout)
#         # self.temporal_fusion = nn.Conv1d(seq_len, 50, kernel_size=1, padding=0, bias=True)
#         # define the pre-fusion subnetworks
#         # 线性层（全连接层），用于将输入特征投影到与隐藏层特征维度相同的空间
#         self.proj = nn.Linear(in_size, hidden_size)
#         self.pooling_mode = "mean"
#         self.d = ensemble_depth
#
#     def merged_strategy(
#             self,
#             hidden_states,
#             mode="mean"
#     ):
#         # 这是一个用于特征融合的方法，根据提供的池化模式对隐藏状态进行融合。可以选择平均、求和或最大池化。
#         if mode == "mean":
#             outputs = torch.mean(hidden_states, dim=1)
#         elif mode == "sum":
#             outputs = torch.sum(hidden_states, dim=1)
#         elif mode == "max":
#             outputs = torch.max(hidden_states, dim=1)[0]
#         else:
#             raise Exception(
#                 "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
#
#         return outputs
#
#     def forward(self, x):
#         '''
#         Args:
#             x: tensor of shape (batch_size, sequence_length, in_size)
#         '''
#         # x = x.transpose(1, 2)  # B N D_in -> B D_in N
#         x = self.proj(x)  # B D_in N - > B D_hidden N (batch_size, sequence_length, hidden_size)
#         # x = x.transpose(1, 2)  # B D_hidden N -> B N D_hidden
#         x, hidden_states = self.backbone(x)
#         # x = self.backbone(x)
#         _x = x
#         if self.d == 1:
#             x = self.merged_strategy(x, self.pooling_mode)
#         else:
#             # l = len(hidden_states)
#             # ensemble_lst = []
#             # for i in range(self.d):
#             #     hidden_state = self.merged_strategy(hidden_states[l - self.d + i], mode=self.pooling_mode)
#             #     ensemble_lst.append(hidden_state)
#             # x = torch.stack(ensemble_lst, dim=1)
#             x = x.mean(dim=1)
#         return x, _x

# V8 ----- transformer fusion
class V8(nn.Module):
    def __init__(self, args):
        super(V8, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_seq_len, self.audio_seq_len, self.video_seq_len = args.seq_lens
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_model = BertTextEncoder(bert_type=args.bert_type, use_finetune=args.use_bert_finetune)
        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim
        self.post_fusion_out = args.post_fusion_out

        self.device = args.device

        # define the pre-fusion subnetworks
        self.tliner = nn.Linear(self.text_in, self.text_hidden)

        # audio and video transformer hyps
        self.audio_nheads = args.sub_nheads
        self.audio_layers = args.sub_layers
        self.video_nheads = args.sub_nheads
        self.video_layers = args.sub_layers
        self.ensemble_depth = args.ensemble_depth

        self.audio_model = MyAVsubNet(self.audio_seq_len,
                                      self.audio_in,
                                      self.audio_hidden,
                                      self.audio_prob,
                                      num_heads=self.audio_nheads,
                                      layers=self.audio_layers,
                                      ensemble_depth=self.ensemble_depth)
        self.video_model = MyAVsubNet(self.video_seq_len,
                                      self.video_in,
                                      self.video_hidden,
                                      self.video_prob,
                                      num_heads=self.video_nheads,
                                      layers=self.video_layers,
                                      ensemble_depth=self.ensemble_depth)

        # define the classify layer for text
        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)

        self.post_text_layer_1 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, 1)

        # define the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_audio_layer_1 = nn.Linear(self.audio_hidden, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, 1)

        # define the classify layer for video
        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.post_video_layer_1 = nn.Linear(self.video_hidden, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, 1)

        # define the classify layer for fusion
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.post_fusion_out, self.post_fusion_out)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_out, self.post_fusion_out)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_out, 1)

        # 定义一个全连接层FC，用于将cls token的信息得到预测结果



        # Output shift
        self.output_range = Parameter(torch.FloatTensor([2]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-1]), requires_grad=False)
        if args.fusion_method == 'LMF':
            self.fusion_network = LMF(self.post_fusion_out,
                                      self.audio_hidden,
                                      self.video_hidden,
                                      self.text_hidden,
                                      device=args.device,
                                      R=6)
        elif args.fusion_method == 'concat':
            self.fusion_network = Concat(self.post_fusion_out,
                                         self.audio_hidden,
                                         self.video_hidden,
                                         self.text_hidden)
        elif args.fusion_method == 'trans':
            self.fusion_network = TransFN(embed_dim=args.post_fusion_out, n_head=args.fus_nheads,
                                          n_layers=args.fus_layers, mlp_ratio=4, drop_prob=0.2)
        else:
            raise NotImplementedError('Fusion method {} not supported!'.format(args.fusion_method))
        self.fusion_method = args.fusion_method

        self.awl = None

    def set_awl(self, awl):
        self.awl = awl

    def set_awl_MICL(self, awl_MICL):
        self.awl_MICL = awl_MICL

    def forward(self, text_x, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
            video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text_h, _text = self.text_model(text_x)
        text_h, _text = self.tliner(text_h), self.tliner(_text)
        audio_h, _audio = self.audio_model(audio_x)
        video_h, _video = self.video_model(video_x)

        # text
        x_t1 = self.post_text_dropout(text_h)
        x_t2 = F.relu(self.post_text_layer_1(x_t1), inplace=True)
        x_t3 = F.relu(self.post_text_layer_2(x_t2), inplace=True)
        output_text = self.post_text_layer_3(x_t3)

        # audio
        x_a1 = self.post_audio_dropout(audio_h)
        x_a2 = F.relu(self.post_audio_layer_1(x_a1), inplace=True)
        # x_a2 = F.relu(self.post_audio_layer_1(audio_h), inplace=True)
        x_a3 = F.relu(self.post_audio_layer_2(x_a2), inplace=True)
        output_audio = self.post_audio_layer_3(x_a3)
        # output_audio = self.post_audio_layer_3(audio_h)
        output_audio = torch.sigmoid(output_audio)
        output_audio = output_audio * self.output_range + self.output_shift

        # video
        x_v1 = self.post_video_dropout(video_h)
        x_v2 = F.relu(self.post_video_layer_1(x_v1), inplace=True)
        # x_v2 = F.relu(self.post_video_layer_1(video_h), inplace=True)
        x_v3 = F.relu(self.post_video_layer_2(x_v2), inplace=True)
        output_video = self.post_video_layer_3(x_v3)
        # output_video = self.post_video_layer_3(video_h)
        output_video = torch.sigmoid(output_video)
        output_video = output_video * self.output_range + self.output_shift

        # multi-modal fusion
        if self.fusion_method == 'trans':
            funsion_ATV, _fusion = self.fusion_network(_audio, _text, _video)
        else:
            funsion_ATV = self.fusion_network(audio_h, text_h, video_h)
        # fusion_data = self.post_fusion_dropout(funsion_ATV)
        # fusion_data = F.relu(self.post_fusion_layer_2(fusion_data), inplace=True)
        # f_v2 = F.relu(self.post_fusion_layer_1(funsion_ATV), inplace=True)
        # f_v3 = F.relu(self.post_fusion_layer_2(f_v2), inplace=True)
        # fusion_output = self.post_fusion_layer_3(f_v3)

        fusion_output = self.post_fusion_layer_3(funsion_ATV)
        # sigmoid -> [0,1] -> [output_shift, output_range + output_shift]
        # 因为情感分析的区间是[-1,1]，所以需要将输出的区间转换到[-1,1]
        output_fusion = torch.sigmoid(fusion_output)
        output_fusion = output_fusion * self.output_range + self.output_shift

        res = {
            'Feature_t': _text,  # (32,50,384)
            'Feature_a': _audio,  # (32,925,384)
            'Feature_v': _video,  # (32,232,384)
            'Fusion': _fusion,  # (32,1207,384)
            'Fusion_token': funsion_ATV,  # (32,384)
            'Feature_t_token': text_h,    # (32,384)
            'Feature_a_token': audio_h,   # (32,384)
            'Feature_v_token': video_h,   # (32,384)
            'M': output_fusion,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res
