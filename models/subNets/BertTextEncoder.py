import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, AlbertModel
from transformers import (
  BertTokenizerFast,
  AutoModel,
)

__all__ = ['BertTextEncoder']

class BertTextEncoder(nn.Module):
    def __init__(self, bert_type='bert', use_finetune=False, ensemble_depth=1):

        super(BertTextEncoder, self).__init__()

        assert bert_type in ['bert', 'albert', 'roberta', 'bert-tiny']

        # directory is fine
        # pretrained_weights = '/home/sharing/disk3/pretrained_embedding/Chinese/bert/pytorch'
        if bert_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('/home/HX/workspace/MSA/MyCode/pretrained_models/bert-base-chinese')
            self.model = BertModel.from_pretrained('/home/HX/workspace/MSA/MyCode/pretrained_models/bert-base-chinese')
        elif bert_type == 'roberta':
            self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
            self.model = BertModel.from_pretrained("clue/roberta_chinese_base")
        elif bert_type == 'albert':
            # albert model
            self.tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_base')
            self.model = AlbertModel.from_pretrained('voidful/albert_chinese_base')
        elif bert_type == 'bert-tiny':
            # albert model
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
            self.model = AutoModel.from_pretrained('ckiplab/bert-tiny-chinese')
            
        self.pooling_mode = 'mean'
        self.use_finetune = use_finetune
        self.d = ensemble_depth
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            # 自集成
            if self.d > 1:
                # Models outputs are now tuples
                outputs = self.model(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids,
                                    output_hidden_states=True)  
                #last hidden state: torch.Size([10, 28, 768])

                hidden_states = []
                for i in range(self.d):
                    hidden_state = outputs[2][12 - i]
                    hidden_state = self.merged_strategy(hidden_state, mode=self.pooling_mode)
                    hidden_states.append(hidden_state)
                hidden_states = torch.stack(hidden_states, dim=1)
                hidden_states = hidden_states.mean(dim=1)

                return hidden_states
            else:
                # Models outputs are now tuples
                outputs = self.model(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids)  
                hidden_states = outputs[0]
                _ = hidden_states
                #last hidden state: torch.Size([10, 28, 768])
                hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
                return hidden_states, _
        else:
            with torch.no_grad():
                
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states

    
    def forward_(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
    
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
    
if __name__ == "__main__":
    bert_normal = BertTextEncoder()
