import torch.nn as nn
import math
import torch
from transformers.models.longformer import LongformerSelfAttention as LSA
from transformers.models.longformer import LongformerConfig

class LongformerSelfAttention(LSA):
    def __init__(self, config_path, layer_id):
        config = LongformerConfig.from_json_file(config_path)
        super(LongformerSelfAttention, self).__init__(config, layer_id)
