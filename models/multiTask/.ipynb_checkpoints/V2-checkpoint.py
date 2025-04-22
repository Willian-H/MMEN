from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder
from models.subNets.FusionNetwork import LMF
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np


__all__ = ['V2']
class AVsubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(AVsubNet, self).__init__()
        # define the pre-fusion subnetworks
        self.proj = nn.Conv1d(in_size, hidden_size, kernel_size=1, padding=0, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_length, in_size)
        '''
        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True,enforce_sorted=False)
        # packed_input=packed_input[0]
        x = x.transpose(1, 2)
        x = self.dropout(self.proj(x))
        x = x.transpose(1, 2)
        # batch_size, sequence_length, hidden_size
        _, (final_h, _) = self.rnn(x)
        lstm_out = torch.cat([final_h[-2], final_h[-1]], -1)
        return lstm_out

class MyAVsubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, num_heads=4, layers=6, ensemble_depth=1):
        super(MyAVsubNet, self).__init__()
        # define the pre-fusion subnetworks
        self.proj = nn.Conv1d(in_size, hidden_size, kernel_size=1, padding=0, bias=False)
        self.backbone = TransformerEncoder(embed_dim=hidden_size, num_heads=num_heads, layers=layers)

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
        x = x.transpose(1, 2)  # B N D_in -> B D_in N
        x = self.proj(x)  # B D_in N - > B D_hidden N
        x = x.permute(2, 0, 1)  # B D_hidden N -> N B D_hidden
        x, hidden_states = self.backbone(x)
        if self.d == 1:
            x = x.permute(1, 0, 2)  # N B D_hidden -> B N D_hidden
            x = self.merged_strategy(x, self.pooling_mode)
        else:
            l = len(hidden_states)
            ensemble_lst = []
            for i in range(self.d):
                hidden_state = self.merged_strategy(hidden_states[l-self.d+i].permute(1,0,2), mode=self.pooling_mode)
                ensemble_lst.append(hidden_state)
            x = torch.stack(ensemble_lst, dim=1)
            x = x.mean(dim=1)
        return x

    
class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3

class V2(nn.Module):
    def __init__(self, args):
        super(V2, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim
        self.post_fusion_out = args.post_fusion_out

        # define the pre-fusion subnetworks
        self.tliner = nn.Linear(self.text_in, self.text_hidden)
        self.audio_model = AVsubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_model = AVsubNet(self.video_in, self.video_hidden, self.video_prob)
        # self.audio_model = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # self.video_model = SubNet(self.video_in, self.video_hidden, self.video_prob)
        # define the classify layer for text
        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)

        self.post_text_layer_1 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, 1)

        # define the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_audio_layer_1 = nn.Linear(self.audio_hidden*2, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, 1)

        # define the classify layer for video
        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.post_video_layer_1 = nn.Linear(self.video_hidden*2, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, 1)

        # define the classify layer for fusion
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.post_text_dim + self.post_audio_dim + self.post_video_dim, self.post_fusion_out)
        # self.post_fusion_layer_1 = nn.Linear(self.post_text_dim, self.post_fusion_out)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_out, self.post_fusion_out)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_out, 1)
        
        # Output shift
        self.output_range = Parameter(torch.FloatTensor([2]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-1]), requires_grad=False)

    def forward(self, text_x, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
            video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text_x = self.text_model(text_x)
        text_h = self.tliner(text_x)
        audio_h = self.audio_model(audio_x)
        video_h = self.video_model(video_x)
        
        # audio_h = self.audio_model(audio_x.squeeze(1))
        # video_h = self.video_model(video_x.squeeze(1))
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
        
        # Transformer fusion
        fusion_cat = torch.cat([x_t3, x_a3, x_v3], dim=-1)
        # fusion_cat = x_t3
        fusion_data = self.post_fusion_dropout(fusion_cat)
        fusion_data = self.post_fusion_layer_1(fusion_data)
        fusion_data = self.post_fusion_layer_2(fusion_data)
        fusion_output = self.post_fusion_layer_3(fusion_data)

        output_fusion = torch.sigmoid(fusion_output)
        output_fusion = output_fusion * self.output_range + self.output_shift

        res = {
            'Feature_t': x_t3,
            'Feature_a': x_a3,
            'Feature_v': x_v3,
            'Feature_f': fusion_data,
            'M': output_fusion ,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res
    
# V3 ------  transformer for audio and visual
class V3(nn.Module):
    def __init__(self, args):
        super(V3, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim
        self.post_fusion_out = args.post_fusion_out

        # define the pre-fusion subnetworks
        self.tliner = nn.Linear(self.text_in, self.text_hidden)
        
        # audio and video transformer hyps
        self.audio_nheads = args.fus_nheads
        self.audio_layers = args.fus_layers
        self.video_nheads = args.fus_nheads
        self.video_layers = args.fus_layers
        self.ensemble_depth = args.ensemble_depth
        
        
        self.audio_model = MyAVsubNet(self.audio_in, 
                                      self.audio_hidden, 
                                      self.audio_prob, 
                                      num_heads=self.audio_nheads, 
                                      layers=self.audio_layers,
                                      ensemble_depth=self.ensemble_depth)
        self.video_model = MyAVsubNet(self.video_in, 
                                      self.video_hidden, 
                                      self.video_prob, 
                                      num_heads=self.video_nheads, 
                                      layers=self.video_layers,
                                      ensemble_depth=self.ensemble_depth)
        # self.audio_model = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # self.video_model = SubNet(self.video_in, self.video_hidden, self.video_prob)
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
        self.post_fusion_layer_1 = nn.Linear(self.post_text_dim + self.post_audio_dim + self.post_video_dim, self.post_fusion_out)
        # self.post_fusion_layer_1 = nn.Linear(self.post_text_dim, self.post_fusion_out)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_out, self.post_fusion_out)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_out, 1)
        
        # Output shift
        self.output_range = Parameter(torch.FloatTensor([2]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-1]), requires_grad=False)
        
        self.w1 = Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.w2 = Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.w3 = Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.w4 = Parameter(torch.FloatTensor([1]), requires_grad=True)

        self.weights = {
            'M': self.w1,
            'T': self.w2,
            'A': self.w3,
            'V': self.w4,
        }
        
    def forward(self, text_x, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
            video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text_x = self.text_model(text_x)
        text_h = self.tliner(text_x)
        audio_h = self.audio_model(audio_x)
        video_h = self.video_model(video_x)
        
        # audio_h = self.audio_model(audio_x.squeeze(1))
        # video_h = self.video_model(video_x.squeeze(1))
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
        
        # Transformer fusion
        fusion_cat = torch.cat([x_t3, x_a3, x_v3], dim=-1)
        # fusion_cat = x_t3
        fusion_data = self.post_fusion_dropout(fusion_cat)
        fusion_data = self.post_fusion_layer_1(fusion_data)
        fusion_data = self.post_fusion_layer_2(fusion_data)
        fusion_output = self.post_fusion_layer_3(fusion_data)

        output_fusion = torch.sigmoid(fusion_output)
        output_fusion = output_fusion * self.output_range + self.output_shift

        res = {
            'Feature_t': x_t3,
            'Feature_a': x_a3,
            'Feature_v': x_v3,
            'Feature_f': fusion_data,
            'M': output_fusion ,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res
    
# V4 ----- low rank factorization
class V4(nn.Module):
    def __init__(self, args):
        super(V4, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
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
        self.audio_nheads = args.fus_nheads
        self.audio_layers = args.fus_layers
        self.video_nheads = args.fus_nheads
        self.video_layers = args.fus_layers
        self.ensemble_depth = args.ensemble_depth
        
        
        self.audio_model = MyAVsubNet(self.audio_in, 
                                      self.audio_hidden, 
                                      self.audio_prob, 
                                      num_heads=self.audio_nheads, 
                                      layers=self.audio_layers,
                                      ensemble_depth=self.ensemble_depth)
        self.video_model = MyAVsubNet(self.video_in, 
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

        self.fusion_network = LMF(self.post_fusion_out,
                                 self.audio_hidden,
                                 self.video_hidden,
                                 self.text_hidden,
                                 device=args.device,
                                 R=6)
    
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
        text_x = self.text_model(text_x)
        text_h = self.tliner(text_x)
        audio_h = self.audio_model(audio_x)
        video_h = self.video_model(video_x)
        
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
        funsion_ATV = self.fusion_network(audio_h, text_h, video_h)
        fusion_data = self.post_fusion_dropout(funsion_ATV)
        fusion_data = F.relu(self.post_fusion_layer_2(fusion_data), inplace=True)
        fusion_output = self.post_fusion_layer_3(fusion_data)
        
        # sigmoid -> [0,1] -> [output_shift, output_range + output_shift]
        output_fusion = torch.sigmoid(fusion_output)
        output_fusion = output_fusion * self.output_range + self.output_shift

        res = {
            'Feature_t': x_t3,
            'Feature_a': x_a3,
            'Feature_v': x_v3,
            'Feature_f': fusion_output,
            'M': output_fusion ,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res
    
# V5 ----- tensor fusion network 
class V5(nn.Module):
    def __init__(self, args):
        super(V5, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
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
        self.audio_nheads = args.fus_nheads
        self.audio_layers = args.fus_layers
        self.video_nheads = args.fus_nheads
        self.video_layers = args.fus_layers
        self.ensemble_depth = args.ensemble_depth
        
        
        self.audio_model = MyAVsubNet(self.audio_in, 
                                      self.audio_hidden, 
                                      self.audio_prob, 
                                      num_heads=self.audio_nheads, 
                                      layers=self.audio_layers,
                                      ensemble_depth=self.ensemble_depth)
        self.video_model = MyAVsubNet(self.video_in, 
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
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_out, self.post_fusion_out)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_out, 1)
        
        # Output shift
        self.output_range = Parameter(torch.FloatTensor([2]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-1]), requires_grad=False)

        # 假设所设秩: R, 期望融合后的特征维度: h
        R, h = 6, self.post_fusion_out
        self.audio_factor = nn.Parameter(torch.Tensor(R, self.audio_hidden + 1, h))
        self.video_factor = nn.Parameter(torch.Tensor(R, self.video_hidden + 1, h))
        self.text_factor = nn.Parameter(torch.Tensor(R, self.text_hidden + 1, h))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, R))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, h))
    
        # init teh factors
        xavier_uniform_(self.audio_factor)
        xavier_uniform_(self.video_factor)
        xavier_uniform_(self.text_factor)
        xavier_uniform_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
    
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
        text_x = self.text_model(text_x)
        text_h = self.tliner(text_x)
        audio_h = self.audio_model(audio_x)
        video_h = self.video_model(video_x)
        
        # audio_h = self.audio_model(audio_x.squeeze(1))
        # video_h = self.video_model(video_x.squeeze(1))
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
        
        n = audio_h.shape[0]
        A = torch.cat([audio_h, torch.ones(n, 1).to(self.device)], dim=1)
        T = torch.cat([text_h, torch.ones(n, 1).to(self.device)], dim=1)
        V = torch.cat([video_h, torch.ones(n, 1).to(self.device)], dim=1)
        # 分解后，并行提取各模态特征
        fusion_A = torch.matmul(A, self.audio_factor)
        fusion_T = torch.matmul(T, self.text_factor)
        fusion_V = torch.matmul(V, self.video_factor)

        # 利用一个Linear再进行特征融合（融合R维度）
        funsion_ATV = fusion_A * fusion_V * fusion_T
        funsion_ATV = torch.matmul(self.fusion_weights, funsion_ATV.permute(1,0,2)).squeeze() + self.fusion_bias

        fusion_data = self.post_fusion_dropout(funsion_ATV)
        fusion_data = self.post_fusion_layer_2(fusion_data)
        fusion_output = self.post_fusion_layer_3(fusion_data)

        output_fusion = torch.sigmoid(fusion_output)
        output_fusion = output_fusion * self.output_range + self.output_shift

        res = {
            'Feature_t': x_t3,
            'Feature_a': x_a3,
            'Feature_v': x_v3,
            'Feature_f': fusion_data,
            'M': output_fusion ,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res