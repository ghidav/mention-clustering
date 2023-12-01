from typing import Any
from transformers import AutoModel, AutoTokenizer
import lightning as L
import torch
import numpy as np

class BERTLikeEncoder(L.LightningModule):

    def __init__(self, model_ckpt, encoder_params) -> None:
        super().__init__()
    
        self.encoder = AutoModel.from_pretrained(model_ckpt, **encoder_params)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.encoder(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
    
    def predict_step(self, batch, batch_id) -> Any:
        input_ids = batch[0]
        attention_mask = batch[1]
        token_type_ids = batch[2]
        return self(input_ids, attention_mask, token_type_ids)
    
class CLSPooling(BERTLikeEncoder):
    # directly use last hidden layer of CLS token
    def __init__(self, model_ckpt, encoder_params) -> None:
        super().__init__(model_ckpt, encoder_params)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return output[:,0,:]

class CLSMeanPooling(BERTLikeEncoder):
    # use mean of CLS hidden layer
    def __init__(self, model_ckpt, encoder_params) -> None:
        super().__init__(model_ckpt, encoder_params)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).hidden_states
        cls_hs = torch.stack(output)[:, :, 0, :] # n_layer * n_batch * 1 * hidden_dim
        return torch.mean(cls_hs, dim=0)

class MeanPooling(BERTLikeEncoder):
    def __init__(self, model_ckpt, encoder_params) -> None:
        super().__init__(model_ckpt, encoder_params)
    
    def mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return self.mean_pooling(output[:,1:,:], attention_mask[:, 1:]) # not use CLS token

class MentionPooling(BERTLikeEncoder):

    def __init__(self, model_ckpt, encoder_params) -> None:
        super().__init__(model_ckpt, encoder_params)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    def mention_mean_pooling(self, last_hidden_state, mention_mask):
        input_mask_expanded = mention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        mention_mask_list = []
        for ex_id, ex in enumerate(input_ids):
            pad_idx = torch.argwhere(ex == self.tokenizer.pad_token_id)[:2].cpu().flatten().numpy() # first two are mention separator
            mention_idx = [pad_idx[0], pad_idx[1]-1]
            retain_idx = list(set(np.arange(ex.shape[0])) - set(pad_idx))
            new_input_ids.append(ex[retain_idx])
            new_attention_mask.append(attention_mask[ex_id][retain_idx])
            new_token_type_ids.append(token_type_ids[ex_id][retain_idx])
            mention_mask = torch.zeros(len(retain_idx)).to(torch.int8).to(ex.device)
            mention_mask[mention_idx[0]:mention_idx[1]] = 1
            mention_mask_list.append(mention_mask)
        input_ids = torch.stack(new_input_ids)
        attention_mask = torch.stack(new_attention_mask)
        token_type_ids = torch.stack(new_token_type_ids)
        mention_mask = torch.stack(mention_mask_list)
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return self.mention_mean_pooling(output, mention_mask)
