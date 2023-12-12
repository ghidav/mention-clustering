from transformers import AutoTokenizer
from tqdm import tqdm


class BERTLikeTokenizer(object):

    def __init__(self, model_ckpt, tokenizer_params, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.tokenizer_params = tokenizer_params

    def build_sentence(self, example):
        return example['left_context'] + ' ' + example['mention'] + ' ' + example['right_context']
    
    def build_sentence_loop(self, examples):
        return [self.build_sentence(ex) for ex in examples]
        
    def tokenize(self, examples):
        return self.tokenizer(self.build_sentence_loop(examples), return_tensors='pt', return_attention_mask=True, **self.tokenizer_params)


class MaskTokenizer(BERTLikeTokenizer):

    def __init__(self, model_ckpt, tokenizer_params, **kwargs):
        super().__init__(model_ckpt, tokenizer_params, **kwargs)

    def build_sentence(self, example):
        if len(example['left_context'].split(' ')) > 30:
            example['left_context'] = ' '.join(example['left_context'].split(' ')[-30:])
        return example['left_context'] + ' ' + self.tokenizer.mask_token + ' ' + example['right_context']
    
class MentionPlusContext(BERTLikeTokenizer):

    def __init__(self, model_ckpt, tokenizer_params, **kwargs):
        super().__init__(model_ckpt, tokenizer_params, **kwargs)

    def build_sentence(self, example):
        if len(example['left_context'].split(' ')) > 30:
            example['left_context'] = ' '.join(example['left_context'].split(' ')[-30:])
        return example['mention'] + self.tokenizer.sep_token + example['left_context'] + self.tokenizer.sep_token + example['right_context']

class PadMention(BERTLikeTokenizer):
    def __init__(self, model_ckpt, tokenizer_params, **kwargs):
        super().__init__(model_ckpt, tokenizer_params, **kwargs)

    def build_sentence(self, example):
        if len(example['left_context'].split(' ')) > 30:
            example['left_context'] = ' '.join(example['left_context'].split(' ')[-30:])
        return example['left_context'] + self.tokenizer.pad_token + example['mention'] + self.tokenizer.pad_token + example['right_context']

class CrossEncoderTokenizer(BERTLikeTokenizer):
    def __init__(self, model_ckpt, tokenizer_params, **kwargs) -> None:
        super().__init__(model_ckpt, tokenizer_params, **kwargs)
        self.max_word_context = tokenizer_params.pop('max_word_context')
        self.tokenizer_params = tokenizer_params

    def truncate_context(self, example):
        if len(example['left_context'].split(' ')) > self.max_word_context:
            example['left_context'] = ' '.join(example['left_context'].split(' ')[-self.max_word_context:])
        if len(example['right_context'].split(' ')) > self.max_word_context:
            example['right_context'] = ' '.join(example['right_context'].split(' ')[:self.max_word_context])
        return example

    def build_sentence(self, ex_a, ex_b):
        ex_a = self.truncate_context(ex_a)
        ex_b = self.truncate_context(ex_b)
        return ex_a['mention'] + self.tokenizer.sep_token + ex_a['left_context'] + self.tokenizer.sep_token + ex_a['right_context'] + \
        ex_b['mention'] + self.tokenizer.sep_token + ex_b['left_context'] + self.tokenizer.sep_token + ex_b['right_context']

    def tokenize(self, ex_a, ex_b):
        return self.tokenizer(self.build_sentence(ex_a, ex_b), return_tensors='pt', return_attention_mask=True, **self.tokenizer_params)