from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from transformers import AutoTokenizer

class BERTTfIdf:
    
    def __init__(self, model_ckpt, tfidf_params, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.tfidf_encoder = TfidfVectorizer(**tfidf_params, vocabulary=self.tokenizer.get_vocab(), tokenizer=self.bert_tokenizer, token_pattern=None)

    def bert_tokenizer(self, text):
        token_id = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.convert_ids_to_tokens(token_id)
    
    def fit(self, examples):
        mentions_string = [ex['mention'] for ex in examples]
        self.tfidf_encoder.fit(mentions_string)
        self.fit = True
    
    def get_embedding(self, examples):
        if self.fit:
            mentions_string = [ex['mention'] for ex in examples]
            scipy_sparse = self.tfidf_encoder.transform(mentions_string)
            coo = scipy_sparse.tocoo()
            values = coo.data
            indices = np.vstack((coo.row, coo.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape
            return torch.sparse_coo_tensor(i, v, torch.Size(shape))
        else:
            raise Exception('TF-IDF not fitted. Please use fit_mention_embedder on your examples list')