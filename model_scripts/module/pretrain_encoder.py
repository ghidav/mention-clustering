import torch
import json
from module.config2class import class_dict
from collections import defaultdict
from lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

class EncoderModule:

    def __init__(self, tokenizer_config, mention_encoder_config, context_encoder_config, split_corpus=False) -> None:
        super().__init__()

        self.tokenizer = class_dict[tokenizer_config['name']](tokenizer_config['model_ckpt'], tokenizer_config['params'])
        self.split_corpus=split_corpus

        self.mention_encoder_config = mention_encoder_config
        
        self.context_encoder = class_dict[context_encoder_config['name']](context_encoder_config['model_ckpt'], context_encoder_config['params'])
        self.batch_size = context_encoder_config['batch_size']
        self.l_trainer = Trainer(**context_encoder_config['trainer_params'], enable_model_summary=None, enable_checkpointing=False, logger=False)

    def import_json(self, json_file):
        examples = []
        with open(json_file, 'r') as f:
            for raw_example in f:
                examples.append(json.loads(raw_example))
        if self.split_corpus:
            examples4corpus = defaultdict(list)
            for ex in examples:
                examples4corpus[ex['corpus']].append(ex)
            self.corpus2id = {name:id for id, name in enumerate(examples4corpus.keys())}
            self.id2corpus = {id:name for id, name in enumerate(examples4corpus.keys())}
            return list(examples4corpus.values())
        else:
            return examples
        
    def istance_mention_encoder(self):
        self.mention_encoder = class_dict[self.mention_encoder_config['name']](self.mention_encoder_config['model_ckpt'], self.mention_encoder_config['params'])
    
    def compute_mention_embedding(self, examples):
        return self.mention_encoder.get_embedding(examples)
    
    def fit_mention_encoder(self, examples):
        self.istance_mention_encoder()
        self.mention_encoder.fit(examples)
    
    def get_mention_embedding(self, examples):
        if self.split_corpus:
            mention_embedding = []
            for examples_corpus in examples:
                self.fit_mention_encoder(examples_corpus)
                mention_embedding.append(self.compute_mention_embedding(examples_corpus))
            return mention_embedding
        else:
            self.fit_mention_encoder(examples)
            return self.compute_mention_embedding(examples) 
    
    def tokenize_examples(self, examples):
        return self.tokenizer.tokenize(examples)
    
    def get_dataloader(self, examples):
        examples_tokenized = self.tokenize_examples(examples)
        dataset = TensorDataset(examples_tokenized['input_ids'], examples_tokenized['attention_mask'], examples_tokenized['token_type_ids'],)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def compute_context_embedding(self, examples):
        dataloader = self.get_dataloader(examples)
        embeddings = self.l_trainer.predict(self.context_encoder, dataloader)
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings
    
    def get_context_embedding(self, examples):
        if self.split_corpus:
            context_embedding = []
            for examples_corpus in examples:
                context_embedding.append(self.compute_context_embedding(examples_corpus))
            return context_embedding
        else:
            return self.compute_context_embedding(examples) 


    

    

