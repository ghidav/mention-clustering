import yaml
from module.pretrain_encoder import EncoderModule
from module.knn_search import KDTreeSearch
import numpy as np
from module.dataset import BiEncoderDataset
from torch.utils.data import DataLoader

with open('/home/dghilardi/mention_clustering/mention-clustering/pytorch_model/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_conf = config['data']
tokenizer_conf = config['tokenizer']
encoder_conf = config['encoder']

encoder = EncoderModule(mention_encoder_config=encoder_conf['mention_encoder'],
                        context_encoder_config=encoder_conf['context_encoder'],
                        tokenizer_config=tokenizer_conf,
                        split_corpus=data_conf['split_corpus'])

val_examples = encoder.import_json(data_conf['file_path']['val'])
val_hockey_ex = [ex for ex in val_examples if ex['corpus']=='ice_hockey']
val_label = np.array([ex['entity_id'] for ex in val_hockey_ex])
# val_context_embedding = encoder.get_context_embedding(val_hockey_ex)
val_mention_embedding = encoder.get_mention_embedding(val_hockey_ex)
# knn_search = KDTreeSearch(X=val_mention_embedding, label=val_label)
# npairs = knn_search.construct_npairs()
val_ex_tokenized = encoder.tokenize_examples(val_examples)
# dataset = BiEncoderDataset(val_mention_embedding, val_ex_tokenized['input_ids'], val_ex_tokenized['attention_mask'],
#                            val_ex_tokenized['token_type_ids'], val_label, npairs)
dataset_b = BiEncoderDataset(val_mention_embedding, val_ex_tokenized['input_ids'], val_ex_tokenized['attention_mask'],
                           val_ex_tokenized['token_type_ids'], val_label)
dataloader = DataLoader(dataset_b, batch_size=4)
x=0
