from torch.utils.data import Dataset
import torch

class BiEncoderDataset(Dataset):

    def __init__(self, sparse_mention_embedding, tokenizer_output, label, ex_ids, npairs=None, label_start_id=0, corpus_id=0) -> None:

        entity2id = {ent:(id+label_start_id) for id, ent in enumerate(set(label))}
        label_encoded = [entity2id[ent_id] for ent_id in label]

        if npairs:
            sparse_mention_embedding_list = []
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            label_list = []
            ex_ids_list =[]
            for npair in npairs:
                sparse_mention_embedding_list.append(torch.stack([sparse_mention_embedding[idx] for idx in npair]))
                input_ids_list.append(torch.stack([tokenizer_output['input_ids'][idx] for idx in npair]))
                attention_mask_list.append(torch.stack([tokenizer_output['attention_mask'][idx] for idx in npair]))
                token_type_ids_list.append(torch.stack([tokenizer_output['token_type_ids'][idx] for idx in npair]))
                label_list.append(torch.tensor([label_encoded[idx] for idx in npair]))
                ex_ids_list.append(torch.tensor([ex_ids[idx] for idx in npair]))
                
            self.sparse_mention_embedding = torch.stack(sparse_mention_embedding_list)
            self.input_ids = torch.stack(input_ids_list).to(torch.int32)
            self.attention_mask = torch.stack(attention_mask_list).to(torch.int8)
            self.token_type_ids = torch.stack(token_type_ids_list).to(torch.int8)
            self.label = torch.stack(label_list).to(torch.int32)
            self.ex_ids = torch.stack(ex_ids_list).to(torch.int32)
            self.corpus = torch.tensor([corpus_id]*len(self.label)).to(torch.int8)
        
        else:
            self.sparse_mention_embedding = sparse_mention_embedding
            self.input_ids = tokenizer_output['input_ids'].to(torch.int32)
            self.attention_mask = tokenizer_output['attention_mask'].to(torch.int8)
            self.token_type_ids = tokenizer_output['token_type_ids'].to(torch.int8)
            self.label = torch.tensor(label_encoded).to(torch.int32)
            self.ex_ids = torch.tensor(ex_ids).to(torch.int32)
            self.corpus = torch.tensor([corpus_id]*len(self.label)).to(torch.int8)
        
        self.label_end_id = max(label_encoded)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        
        return {'sparse_mention_embedding': self.sparse_mention_embedding[index],
                'input_ids': self.input_ids[index],
                'attention_mask': self.attention_mask[index],
                'token_type_ids': self.token_type_ids[index],
                'label': self.label[index],
                'corpus':self.corpus[index],
                'ex_id': self.ex_ids[index]}

class BiEncoderDatasetWithMentionMask(Dataset):

    def __init__(self, sparse_mention_embedding, tokenizer_output, label, ex_ids, npairs=None, label_start_id=0, corpus_id=0) -> None:

        entity2id = {ent:(id+label_start_id) for id, ent in enumerate(set(label))}
        label_encoded = [entity2id[ent_id] for ent_id in label]

        if npairs:
            sparse_mention_embedding_list = []
            input_ids_list = []
            attention_mask_list = []
            mention_mask_list = []
            token_type_ids_list = []
            label_list = []
            ex_ids_list =[]
            for npair in npairs:
                sparse_mention_embedding_list.append(torch.stack([sparse_mention_embedding[idx] for idx in npair]))
                input_ids_list.append(torch.stack([tokenizer_output['input_ids'][idx] for idx in npair]))
                attention_mask_list.append(torch.stack([tokenizer_output['attention_mask'][idx] for idx in npair]))
                mention_mask_list.append(torch.stack([tokenizer_output['mention_mask'][idx] for idx in npair]))
                token_type_ids_list.append(torch.stack([tokenizer_output['token_type_ids'][idx] for idx in npair]))
                label_list.append(torch.tensor([label_encoded[idx] for idx in npair]))
                ex_ids_list.append(torch.tensor([ex_ids[idx] for idx in npair]))
                
            self.sparse_mention_embedding = torch.stack(sparse_mention_embedding_list)
            self.input_ids = torch.stack(input_ids_list).to(torch.int32)
            self.attention_mask = torch.stack(attention_mask_list).to(torch.int8)
            self.mention_mask = torch.stack(mention_mask_list).to(torch.int8)
            self.token_type_ids = torch.stack(token_type_ids_list).to(torch.int8)
            self.label = torch.stack(label_list).to(torch.int32)
            self.ex_ids = torch.stack(ex_ids_list).to(torch.int32)
            self.corpus = torch.tensor([corpus_id]*len(self.label)).to(torch.int8)
        
        else:
            self.sparse_mention_embedding = sparse_mention_embedding
            self.input_ids = tokenizer_output['input_ids'].to(torch.int32)
            self.attention_mask = tokenizer_output['attention_mask'].to(torch.int8)
            self.mention_mask = tokenizer_output['mention_mask'].to(torch.int8)
            self.token_type_ids = tokenizer_output['token_type_ids'].to(torch.int8)
            self.label = torch.tensor(label_encoded).to(torch.int32)
            self.ex_ids = torch.tensor(ex_ids).to(torch.int32)
            self.corpus = torch.tensor([corpus_id]*len(self.label)).to(torch.int8)
        
        self.label_end_id = max(label_encoded)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        
        return {'sparse_mention_embedding': self.sparse_mention_embedding[index],
                'input_ids': self.input_ids[index],
                'attention_mask': self.attention_mask[index],
                'mention_mask': self.mention_mask[index],
                'token_type_ids': self.token_type_ids[index],
                'label': self.label[index],
                'corpus':self.corpus[index],
                'ex_id': self.ex_ids[index]}

class CrossEncoderDataset(Dataset):

    def __init__(self, examples) -> None:
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]