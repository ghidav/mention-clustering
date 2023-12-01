import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from module.config2class import class_dict
from collections import defaultdict
import json
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import pickle
from pathlib import Path

class BiEncoderDataModule(L.LightningDataModule):

    def __init__(self, data_config, tokenizer_config, mention_encoder_config, knn_config) -> None:

        self.split_corpus = data_config['split_corpus']
        self.file_path = data_config['file_path']
        self.data_mode = data_config['mode']
        self.batch_train = data_config['batch']['train']
        self.batch_test = data_config['batch']['test']

        self.output_path = Path(data_config['output_path'])
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.tokenizer = class_dict[tokenizer_config['name']](tokenizer_config['model_ckpt'], tokenizer_config['params'])

        self.mention_encoder_config = mention_encoder_config

        self.knn_config = knn_config

        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self._log_hyperparams = True

        self.dataset_module = class_dict[data_config['dataset_name']]

    def import_json(self, json_file):
        examples = []
        with open(json_file, 'r') as f:
            for ex_id, raw_example in enumerate(f):
                ex = json.loads(raw_example)
                ex['ex_id'] = ex_id
                examples.append(ex)
        if self.split_corpus:
            examples4corpus = defaultdict(list)
            for ex in examples:
                examples4corpus[ex['corpus']].append(ex)
            corpus2id = {name:id for id, name in enumerate(examples4corpus.keys())}
            return list(examples4corpus.values()), corpus2id
        else:
            return [examples], {'no_corupus':0}
    
    def istance_mention_encoder(self):
        self.mention_encoder = class_dict[self.mention_encoder_config['name']](self.mention_encoder_config['model_ckpt'], self.mention_encoder_config['params'])
    
    def compute_mention_embedding(self, examples):
        return self.mention_encoder.get_embedding(examples)
    
    def fit_mention_encoder(self, examples):
        self.istance_mention_encoder()
        self.mention_encoder.fit(examples)
    
    def get_mention_embedding(self, examples):
        self.fit_mention_encoder(examples)
        mention_embedding = self.compute_mention_embedding(examples)
        return mention_embedding

    def tokenize_examples(self, examples):
        return self.tokenizer.tokenize(examples)
    
    
    def build_dataset(self, examples, label_start_id=0, predict=False, shuffle=False, corpus_id=0):
        labels = np.array([ex['entity_id'] for ex in examples])
        ex_ids = np.array([ex['ex_id'] for ex in examples])
        mention_embeddings = self.get_mention_embedding(examples)
        examples_tokenized = self.tokenize_examples(examples)

        # return dataset with single examples (biencoder in predict mode)
        if predict:
            return self.dataset_module(mention_embeddings,
                                    examples_tokenized,
                                    labels,
                                    ex_ids, 
                                    label_start_id=label_start_id,
                                    corpus_id=corpus_id)
        
        # return dataset with tuple of n-element 
        # with anchor point and m-positves and k-negatives examples (biencoder in training mode)
        else:
            knn_searcher = class_dict[self.knn_config['name']](self.knn_config['hard_pairs']['fit_params'])
            npairs = knn_searcher.construct_npairs(mention_embeddings, labels, **self.knn_config['hard_pairs']['query_params'])
            if shuffle:
                # shuffle for traing dataset
                np.random.shuffle(npairs)
            return self.dataset_module(mention_embeddings,
                                    examples_tokenized,
                                    labels,
                                    ex_ids,
                                    npairs, 
                                    label_start_id=label_start_id)
        
    def prepare_data(self):
        if self.data_mode == 'Create':
            dataset_dict = {}
            corpus_dict = {}
            for data_partition, file_path in self.file_path.items():
                print(f'Create {data_partition} dataset')
                label_start_id=0
                dataset_list = []
                examples, corpus2id = self.import_json(file_path)

                if data_partition=='train':
                    for corpus_id, examples_corpus in enumerate(examples):
                        dataset = self.build_dataset(examples_corpus, 
                                                    label_start_id=label_start_id,
                                                    predict=False,
                                                    corpus_id=corpus_id)
                        dataset_list.append(dataset)
                        label_start_id = dataset.label_end_id + 1
                        
                elif data_partition=='pred' or data_partition=='test' or data_partition=='val':
                    for corpus_id, examples_corpus in enumerate(examples):
                        dataset = self.build_dataset(examples_corpus, 
                                                    label_start_id=label_start_id,
                                                    predict=True,
                                                    shuffle=False,
                                                    corpus_id=corpus_id)
                        dataset_list.append(dataset)
                        label_start_id = dataset.label_end_id + 1
                
                # merge corpus dataset in a single dataset
                dataset = ConcatDataset(dataset_list)
                dataset_dict[data_partition] = dataset
                corpus_dict[data_partition] = corpus2id
            
            # save file to disk
            with open(self.output_path / 'dataset.pkl', 'wb') as f_out:
                pickle.dump(dataset_dict, f_out)
            with open(self.output_path / 'corpus2dict.pkl', 'wb') as f_out:
                pickle.dump(corpus_dict, f_out)
        
        if self.data_mode == 'Load':
            pass
    
    def setup(self, stage: str=''):
        
        # load dataset
        with open(self.output_path / 'dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        
        if stage=='fit' or stage=='' or stage=='validate':
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['val']
        
        elif stage=='test':
            self.test_dataset = dataset['test']
        
        elif stage=='predict':
            self.predict_dataset = dataset['pred']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_train, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_test, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_test, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_test, shuffle=False)