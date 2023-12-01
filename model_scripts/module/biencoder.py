from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from module.config2class import class_dict
import torch
from collections import defaultdict
from itertools import combinations, product
from scipy.sparse import csr_matrix

class BiEncoder(L.LightningModule):

    def __init__(self, contex_encoder_config, optmizer_config, loss_config, miner_config, knn_config):
        super().__init__()
        self.context_encoder = class_dict[contex_encoder_config['name']](contex_encoder_config['model_ckpt'], contex_encoder_config['params'])
        self.optimizer_config = optmizer_config
        self.loss = class_dict[loss_config['name']](loss_config['params'])

        if miner_config:
            self.miner = class_dict[miner_config['name']](miner_config['params'])
        else:
            self.miner = None
        
        self.projector = torch.nn.LazyLinear(out_features=128, bias=True)
        
        self.knn_searcher = class_dict[knn_config['name']](knn_config['biencoder_evaluation']['fit_params'])

        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)
        self.pred_output = defaultdict(list)
    
    def flat_batch(self, batch):
        n_batch, n_pairs, n_tokens = batch['input_ids'].shape
        reshape_dim = n_batch*n_pairs
        batch['input_ids'] = batch['input_ids'].view(reshape_dim, -1)
        batch['attention_mask'] = batch['attention_mask'].view(reshape_dim, -1)
        batch['token_type_ids'] = batch['token_type_ids'].view(reshape_dim, -1)
        batch['sparse_mention_embedding'] = batch['sparse_mention_embedding'].to_dense().view(reshape_dim, -1)
        batch['label'] = batch['label'].view(reshape_dim)
        batch['ex_id'] = batch['ex_id'].view(reshape_dim)
        return batch
    
    def forward(self, batch):
        input_ids = batch['input_ids'].to(torch.int32)
        attention_mask = batch['attention_mask'].to(torch.int32)
        token_type_ids = batch['token_type_ids'].to(torch.int32)
        mention_embedding = batch['sparse_mention_embedding'].to_dense().to(torch.int32)
        context_embedding = self.context_encoder(input_ids, attention_mask, token_type_ids)
        mention_embedding = mention_embedding
        full_embedding = torch.cat([mention_embedding, context_embedding], dim=1)
        hidden_embedding = self.projector(full_embedding)
        return hidden_embedding
    
    def training_step(self, batch, batch_idx):
        batch = self.flat_batch(batch)
        embeddings = self(batch)
        lables = batch['label'].to(torch.int32)
        if self.miner:
            miner_pairs = self.miner.get_pairs([embeddings, lables])
        else:
            miner_pairs = None
        loss = self.loss.compute_loss([embeddings, lables, miner_pairs])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        embeddings = self(batch)
        lables = batch['label'].to(torch.int32)
        corpus = batch['corpus']
        ex_ids = batch['ex_id']
        self.val_output['embeddings'].append(embeddings)
        self.val_output['lables'].append(lables)
        self.val_output['corpus'].append(corpus)
        self.val_output['ex_id'].append(ex_ids)
    
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            pass
        else:
            embeddings = torch.cat(self.val_output['embeddings'], dim=0)
            labels = torch.cat(self.val_output['lables'], dim=0)
            corpus = torch.cat(self.val_output['corpus'], dim=0)
            ex_id = torch.cat(self.val_output['ex_id'], dim=0)
            positive_pairs = self.get_positive_pairs(labels, corpus, ex_id)
            neighbors_pairs = self.get_neighbors_pairs(embeddings, corpus, ex_id)
            recall10 = len(set(positive_pairs) & set(neighbors_pairs)) / len(set(positive_pairs))
            self.log('val_recall10', recall10)
            self.val_output.clear()
    
    def test_step(self, batch, batch_idx):
        embeddings = self(batch)
        lables = batch['label'].to(torch.int32)
        corpus = batch['corpus']
        ex_ids = batch['ex_id']
        self.test_output['embeddings'].append(embeddings)
        self.test_output['lables'].append(lables)
        self.test_output['corpus'].append(corpus)
        self.test_output['ex_id'].append(ex_ids)
    
    def on_test_epoch_end(self):
        embeddings = torch.cat(self.test_output['embeddings'], dim=0)
        labels = torch.cat(self.test_output['lables'], dim=0)
        corpus = torch.cat(self.test_output['corpus'], dim=0)
        ex_id = torch.cat(self.test_output['ex_id'], dim=0)
        positive_pairs = self.get_positive_pairs(labels, corpus, ex_id)
        neighbors_pairs = self.get_neighbors_pairs(embeddings, corpus, ex_id)
        recall10 = len(set(positive_pairs) & set(neighbors_pairs)) / len(set(positive_pairs))
        self.log('test_recall10', recall10)
        self.test_output.clear()

    def predict_step(self, batch, batch_idx):
        embeddings = self(batch)
        sparse_mention_embedding = batch['sparse_mention_embedding']
        lables = batch['label'].to(torch.int32)
        corpus = batch['corpus']
        ex_ids = batch['ex_id']
        self.pred_output['embeddings'].append(embeddings)
        self.pred_output['sparse_mention_embedding'].append(sparse_mention_embedding)
        self.pred_output['lables'].append(lables)
        self.pred_output['corpus'].append(corpus)
        self.pred_output['ex_id'].append(ex_ids)
    
    def on_predict_epoch_end(self):
        embeddings = torch.cat(self.pred_output['embeddings'], dim=0)
        sparse_mention_embedding = torch.cat(self.pred_output['sparse_mention_embedding'], dim=0)
        labels = torch.cat(self.pred_output['lables'], dim=0)
        corpus = torch.cat(self.pred_output['corpus'], dim=0)
        ex_id = torch.cat(self.pred_output['ex_id'], dim=0)
        positive_pairs = self.get_positive_pairs(labels, corpus, ex_id)
        positive_pairs = set(positive_pairs)
        neighbors_pairs = self.get_neighbors_pairs(embeddings, corpus, ex_id)
        neighbors_pairs = set(neighbors_pairs)
        negative_pairs = neighbors_pairs - positive_pairs
        ordered_pairs = list(positive_pairs) + list(negative_pairs)
        orderd_link_labels = [1]*len(positive_pairs) + [0]*len(negative_pairs)
        self.pred_pairs = {'pairs':ordered_pairs, 'link':orderd_link_labels}
        self.pred_dict_output = {'biencoder_embedding':embeddings,
                                 'mention_embedding':sparse_mention_embedding,
                                 'labels':labels,
                                 'corpus':corpus,
                                 'ex_id':ex_id}
 
    def get_positive_pairs(self, labels, corpus, ex_id):
        positive_pairs_list = []
        for corpus_id in torch.unique(corpus):
            labels_corpus = labels[corpus==corpus_id].cpu().numpy()
            ex_id_corpus = ex_id[corpus==corpus_id].cpu().numpy()
            label2ex, _ = self.knn_searcher.get_positive_pairs(labels_corpus)
            for examples in label2ex.values():
                if len(examples) > 1:
                    examples = [ex_id_corpus[idx] for idx in examples]
                    positive_pairs = list(combinations(examples, 2))
                    positive_pairs_list += [tuple(sorted(pair)) for pair in positive_pairs]
        return positive_pairs_list
    
    def get_neighbors_pairs(self, embeddings, corpus, ex_id):
        neighbors_pairs_list = []
        for corpus_id in torch.unique(corpus):
            embeddings_corpus = embeddings[corpus==corpus_id].cpu().numpy()
            ex_id_corpus = ex_id[corpus==corpus_id].cpu().numpy()
            self.knn_searcher.fit(embeddings_corpus)
            for embd_id, embd in enumerate(embeddings_corpus):
                neighbors_idx, dist = self.knn_searcher.query(embd.reshape(1,-1), k=11)
                neighbors_idx = list(set(neighbors_idx[0].tolist()) - set([embd_id]))
                neighbors_idx = [ex_id_corpus[idx] for idx in neighbors_idx]
                neighbors_pairs = list(product(neighbors_idx, [ex_id_corpus[embd_id]]))
                neighbors_pairs_list += [tuple(sorted(pair)) for pair in neighbors_pairs]
        return neighbors_pairs_list

    def configure_optimizers(self):
        context_encoder_param = list(self.context_encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in context_encoder_param
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in context_encoder_param
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0}]
            
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=float(self.optimizer_config['learning_rate']))

class BiEncoderNoMention(BiEncoder):

    def __init__(self, contex_encoder_config, optmizer_config, loss_config, miner_config, knn_config):
        super().__init__(contex_encoder_config, optmizer_config, loss_config, miner_config, knn_config)
        self.projector = None
    
    def forward(self, batch):
        input_ids = batch['input_ids'].to(torch.int32)
        attention_mask = batch['attention_mask'].to(torch.int32)
        token_type_ids = batch['token_type_ids'].to(torch.int32)
        context_embedding = self.context_encoder(input_ids, attention_mask, token_type_ids)
        return context_embedding


class BiEncoderOnlyMention(BiEncoder):

    def __init__(self, contex_encoder_config, optmizer_config, loss_config, miner_config, knn_config):
        super().__init__(contex_encoder_config, optmizer_config, loss_config, miner_config, knn_config)
    
    def forward(self, batch):
        return batch['sparse_mention_embedding']
    
    def get_neighbors_pairs(self, embeddings, corpus, ex_id):
        neighbors_pairs_list = []
        for corpus_id in torch.unique(corpus):
            embeddings_corpus = embeddings.to_dense()[corpus==corpus_id].cpu().numpy()
            embeddings_corpus = csr_matrix(embeddings_corpus)
            ex_id_corpus = ex_id[corpus==corpus_id].cpu().numpy()
            self.knn_searcher.fit(embeddings_corpus)
            for embd_id, embd in enumerate(embeddings_corpus):
                neighbors_idx, dist = self.knn_searcher.query(embd.reshape(1,-1), k=11)
                neighbors_idx = list(set(neighbors_idx[0].tolist()) - set([embd_id]))
                neighbors_idx = [ex_id_corpus[idx] for idx in neighbors_idx]
                neighbors_pairs = list(product(neighbors_idx, [ex_id_corpus[embd_id]]))
                neighbors_pairs_list += [tuple(sorted(pair)) for pair in neighbors_pairs]
        return neighbors_pairs_list