from torch.nn.init import xavier_uniform_
import torch.nn as nn
import torch

class Concat(nn.Module):
    def __init__(self, post_fusion_out, post_audio_dim, post_video_dim, post_text_dim):
        super(Concat, self).__init__()
        self.post_fusion_layer_1 = nn.Linear(post_text_dim + post_audio_dim + post_video_dim, post_fusion_out)

    def forward(self, audio_h, text_h, video_h):
        fusion = torch.cat([audio_h, text_h, video_h], dim=-1)
        fusion = self.post_fusion_layer_1(fusion)
        return fusion
    
class LMF(nn.Module):
    def __init__(self, fusion_dim, audio_hidden, video_hidden, text_hidden, device, R=6):
        super(LMF, self).__init__()
        # 假设所设秩: R, 期望融合后的特征维度: h
        self.R, self.h = R, fusion_dim
        self.audio_factor = nn.Parameter(torch.Tensor(self.R, audio_hidden + 1, self.h))
        self.video_factor = nn.Parameter(torch.Tensor(self.R, video_hidden + 1, self.h))
        self.text_factor = nn.Parameter(torch.Tensor(self.R, text_hidden + 1, self.h))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.R))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.h))
    
        # init teh factors
        xavier_uniform_(self.audio_factor)
        xavier_uniform_(self.video_factor)
        xavier_uniform_(self.text_factor)
        xavier_uniform_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        
        self.device = device
        
    def forward(self, audio_h, text_h, video_h):
        '''
        Args:
            audio_h: tensor of shape (batch_size, audio_hidden)
            text_h: tensor of shape (batch_size, text_hidden)
            video_h: tensor of shape (batch_size, video_hidden)
        '''
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

        return funsion_ATV
            
from models.subNets.transformers_encoder.my_transformer import EncoderLayer

class TransFN(nn.Module):
    def __init__(self, embed_dim, n_head, n_layers, mlp_ratio=4, drop_prob=0.1, modal_embeded=True):
        super(TransFN, self).__init__()
        
        # self.audio_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.video_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.text_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modal_type_embeddings = nn.Embedding(3, embed_dim) if modal_embeded else None
        self.audio_type_idx = 0
        self.video_type_idx = 1
        self.text_type_idx = 2
        self.modal_embeded = modal_embeded
        # self.video_modal_embed = nn.Parameter(torch.randn(1, embed_len, d_model) * .02)
        # self.text_modal_embed = nn.Parameter(torch.randn(1, embed_len, d_model) * .02)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=embed_dim,
                                                  ffn_hidden=embed_dim*mlp_ratio,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
    def forward(self, audio_h, text_h, video_h, s_mask=None):
        '''
        Args:
            audio_h: tensor of shape (batch_size, audio_seq_len, audio_hidden)
            text_h: tensor of shape (batch_size, text_seq_len, text_hidden)
            video_h: tensor of shape (batch_size, video_seq_len, video_hidden)
        '''
        if self.modal_embeded:
            audio_h = audio_h + self.modal_type_embeddings(torch.full_like(audio_h[:,:,0], self.audio_type_idx).long())
            video_h = video_h + self.modal_type_embeddings(torch.full_like(video_h[:,:,0], self.video_type_idx).long())
            text_h = text_h + self.modal_type_embeddings(torch.full_like(text_h[:,:,0], self.text_type_idx).long())
        
        co_embeds = torch.cat([audio_h, text_h, video_h], dim=1)
        co_embeds = self._cls_embed(co_embeds)
        for layer in self.layers:
            x = layer(co_embeds, s_mask)
        # output cls token and the rest
        return x[:,0,:], x[:,1:,:]
    
    def _cls_embed(self, x):
        # pos_embed has entry for class token, concat then add
        if self.fusion_cls_token is not None:
            x = torch.cat((self.fusion_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x

    

    
from models.subNets.transformers_encoder.transformer import TransformerEncoder
class MulT(nn.Module):
    def __init__(self, n_head, n_layers, orig_d_a, orig_d_v, orig_d_l, drop_prob=0.1):
        super(MulT, self).__init__()
        self.d_a = 30
        self.d_v = 30
        self.d_l = 30
        self.lonly = self.aonly = self.vonly = True
        self.attn_dropout = drop_prob
        self.attn_dropout_a = drop_prob
        self.attn_dropout_v = drop_prob
        self.num_heads = n_head
        self.layers = n_layers
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = 0.1
        self.embed_dropout = 0.1
        self.attn_mask = False
        
        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def forward(self, audio_h, text_h, video_h, s_mask=None):
        '''
        Args:
            audio_h: tensor of shape (batch_size, audio_seq_len, audio_hidden)
            text_h: tensor of shape (batch_size, text_seq_len, text_hidden)
            video_h: tensor of shape (batch_size, video_seq_len, video_hidden)
        '''
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(text_h)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(audio_h)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(video_h)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs
    
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)