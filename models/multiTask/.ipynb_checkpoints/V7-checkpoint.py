from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder
from models.subNets.transformers_encoder.transformer_vit import Encoder

from models.subNets.FusionNetwork import LMF
from models.subNets.FusionNetwork import Concat
from models.subNets.FusionNetwork import TransFN
import numpy as np


__all__ = ['V7']
class MyAVsubNet(nn.Module):
    def __init__(self, seq_len, in_size, hidden_size, dropout, num_heads=4, layers=6, ensemble_depth=1):
        super(MyAVsubNet, self).__init__()

        self.backbone = Encoder(d_model=hidden_size, embed_len=seq_len, n_head=num_heads, n_layers=layers, drop_prob=dropout)
        # self.temporal_fusion = nn.Conv1d(seq_len, 50, kernel_size=1, padding=0, bias=True)
        # define the pre-fusion subnetworks
        self.proj = nn.Linear(in_size, hidden_size)
        self.pooling_mode = "mean"
        self.d = ensemble_depth
        
    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
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
        _x = x
        if self.d == 1:
            x = self.merged_strategy(x, self.pooling_mode)
        else:
            l = len(hidden_states)
            ensemble_lst = []
            for i in range(self.d):
                hidden_state = self.merged_strategy(hidden_states[l-self.d+i], mode=self.pooling_mode)
                ensemble_lst.append(hidden_state)
            x = torch.stack(ensemble_lst, dim=1)
            x = x.mean(dim=1)
        return x, _x



    
# V7 ----- transformer fusion
class V7(nn.Module):
    def __init__(self, args):
        super(V7, self).__init__()
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
            self.fusion_network = TransFN(embed_dim=args.post_fusion_out, n_head=args.fus_nheads, n_layers=args.fus_layers, mlp_ratio=4, drop_prob=0.2)
        else:
            raise NotImplementedError('Fusion method {} not supported!'.format(args.fusion_method))
        self.fusion_method = args.fusion_method
            
        self.awl = None
    
    def set_awl(self, awl):
        self.awl = awl
        

    
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
        x_a3 = F.relu(self.post_audio_layer_2(x_a2), inplace=True)
        output_audio = self.post_audio_layer_3(x_a3)
        
        # video
        x_v1 = self.post_video_dropout(video_h)
        x_v2 = F.relu(self.post_video_layer_1(x_v1), inplace=True)
        x_v3 = F.relu(self.post_video_layer_2(x_v2), inplace=True)
        output_video = self.post_video_layer_3(x_v3)
        
        # multi-modal fusion
        if self.fusion_method == 'trans':
            funsion_ATV = self.fusion_network(_audio, _text, _video)
        else:
            funsion_ATV = self.fusion_network(audio_h, text_h, video_h)
        # fusion_data = self.post_fusion_dropout(funsion_ATV)
        # fusion_data = F.relu(self.post_fusion_layer_2(fusion_data), inplace=True)
        fusion_output = self.post_fusion_layer_3(funsion_ATV)
        
        # sigmoid -> [0,1] -> [output_shift, output_range + output_shift]
        output_fusion = torch.sigmoid(fusion_output)
        output_fusion = output_fusion * self.output_range + self.output_shift

        res = {
            'M': output_fusion ,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res
