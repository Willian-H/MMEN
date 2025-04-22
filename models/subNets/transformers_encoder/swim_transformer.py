import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_heads(self.q_linear(query), batch_size)
        key = self.split_heads(self.k_linear(key), batch_size)
        value = self.split_heads(self.v_linear(value), batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e4"))

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, value)

        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.output_linear(context)

        return output, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x


class SWIMTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(SWIMTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        hidden_states = []
        for layer in self.layers:
            x = layer(x, mask)
            hidden_states.append(x)  # 将每一层的输出保存在列表中
        return x, hidden_states  # 返回最终的输出和中间的hidden_states


# # Parameters
# embed_dim = 512
# num_heads = 8
# ff_hidden_dim = 2048
# num_layers = 6
# dropout = 0.1
#
# # Creating the SWIM Transformer Encoder
# encoder = SWIMTransformerEncoder(embed_dim, num_heads, ff_hidden_dim, num_layers, dropout)
#
# # Example input
# seq_length = 100
# batch_size = 32
# input_data = torch.randn(batch_size, seq_length, embed_dim)
#
# # Example mask (you might want to create a proper mask depending on your use case)
# mask = torch.ones(batch_size, seq_length)
#
# # Getting encoder output
# encoder_output = encoder(input_data, mask)
