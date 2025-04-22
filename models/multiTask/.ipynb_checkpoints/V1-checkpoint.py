from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
__all__ = ['V1']
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
    def __init__(self, in_size, hidden_size, dropout):
        super(MyAVsubNet, self).__init__()
        # define the pre-fusion subnetworks
        self.proj = nn.Conv1d(in_size, hidden_size, kernel_size=1, padding=0, bias=False)
        self.backbone = TransformerEncoder(embed_dim=hidden_size, num_heads=4, layers=6)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_length, in_size)
        '''
        # B N D_in -> B D_in N
        x = x.transpose(1, 2)
        # B D_in N - > B D_hidden N
        x = self.proj(x)
        # B D_hidden N -> N B D_hidden
        x = x.permute(2, 0, 1)
        x = self.backbone(x)
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

class V1(nn.Module):
    def __init__(self, args):
        super(V1, self).__init__()
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
        # self.audio_model = AVsubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # self.video_model = AVsubNet(self.video_in, self.video_hidden, self.video_prob)
        self.audio_model = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_model = SubNet(self.video_in, self.video_hidden, self.video_prob)
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

    def forward(self, text_x, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text_x = self.text_model(text_x)[:,0,:]
        text_h = self.tliner(text_x)
        
        audio_h = self.audio_model(audio_x.squeeze(1))
        video_h = self.video_model(video_x.squeeze(1))
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

    def forward(self, text_x, audio_x, video_x, a_len, v_len):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)  bs, 232, 177
            video_x: tensor of shape (batch_size, sequence_len, video_in)  bs, 925, 25
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text_x = self.text_model(text_x)[:,0,:]
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