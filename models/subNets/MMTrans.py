import torch.nn as nn
import torch
            
from models.subNets.transformers_encoder.my_transformer import EncoderLayer

class MMTrans(nn.Module):
    def __init__(self, args, mlp_ratio=4, drop_prob=0.1):
        super(MMTrans, self).__init__()
        self.text_seq_len, self.audio_seq_len, self.video_seq_len = args.seq_lens
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.nheads = args.fus_nheads
        self.layers = args.fus_layers
        embed_dim = 768
        self.embed_dim = embed_dim
        
        self.audio_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.video_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.audio_pos_embed = nn.Parameter(torch.randn(1, self.audio_seq_len, embed_dim) * .02)
        self.video_pos_embed = nn.Parameter(torch.randn(1, self.video_seq_len, embed_dim) * .02)

        
        # self.pos_embed = nn.Parameter(torch.randn(1, embed_len, d_model) * .02)
        self.proj_t = nn.Conv1d(self.text_in, embed_dim, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.audio_in, embed_dim, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.video_in, embed_dim, kernel_size=1, padding=0, bias=False)

        self.layers = nn.ModuleList([EncoderLayer(d_model=embed_dim,
                                                  ffn_hidden=embed_dim*mlp_ratio,
                                                  n_head=self.nheads,
                                                  drop_prob=drop_prob)
                                     for _ in range(self.layers)])
        
    def forward(self, audio_h, text_h, video_h, s_mask=None):
        '''
        Args:
            audio_h: tensor of shape (batch_size, audio_seq_len, audio_hidden)
            text_h: tensor of shape (batch_size, text_seq_len, text_hidden)
            video_h: tensor of shape (batch_size, video_seq_len, video_hidden)
        '''
        # proj to (batch_size, seq_len, embed_dim)
        audio_h = audio_h if audio_h.size(2) == self.embed_dim else self.proj_a(audio_h.transpose(1,2)).transpose(1,2)
        text_h = text_h if text_h.size(2) == self.embed_dim else self.proj_t(text_h.transpose(1,2)).transpose(1,2)
        video_h = video_h if video_h.size(2) == self.embed_dim else self.proj_v(video_h.transpose(1,2)).transpose(1,2)

        co_embeds = torch.cat([audio_h, text_h, video_h], dim=1)
        co_embeds = self._cls_embed(co_embeds)
        for layer in self.layers:
            x = layer(co_embeds, s_mask)
        # output cls token
        # M A T V
        return x[:,0,:],x[:,1,:],x[:,2,:],x[:,3,:]
    
    def _cls_embed(self, x):
        # pos_embed has entry for class token, concat then add
        if self.fusion_cls_token is not None:
            x = torch.cat((self.fusion_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.audio_cls_token is not None:
            x = torch.cat((self.audio_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.text_cls_token is not None:
            x = torch.cat((self.text_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.video_cls_token is not None:
            x = torch.cat((self.video_cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        return x
    
    def _pos_embed(self, x):
        # pos_embed has entry for class token, concat then add
        if self.fusion_cls_token is not None:
            x = torch.cat((self.fusion_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.audio_cls_token is not None:
            x = torch.cat((self.audio_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.text_cls_token is not None:
            x = torch.cat((self.text_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.video_cls_token is not None:
            x = torch.cat((self.video_cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        return x