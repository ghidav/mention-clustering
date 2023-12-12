from module.mention_encoder import BERTTfIdf
from module.tokenizer import BERTLikeTokenizer, MaskTokenizer, MentionPlusContext, PadMention, CrossEncoderTokenizer
from module.PLM_encoder import CLSPooling, BERTLikeEncoder, MeanPooling, CLSMeanPooling, MentionPooling
from module.knn_search import KDTreeSearch, NNDescentSearch
from module.losses import MSLoss, BinaryCrossEntropyLoss
from module.miners import MSMiner
from module.dataset import BiEncoderDataset, CrossEncoderDataset


class_dict = {
    'BERTTfIdf':BERTTfIdf,
    'BERTLikeTokenizer':BERTLikeTokenizer,
    'CLSPooling':CLSPooling,
    'BERTLikeEncoder':BERTLikeEncoder,
    'KDTreeSearch':KDTreeSearch,
    'MSLoss':MSLoss,
    'MSMiner':MSMiner,
    'MaskTokenizer':MaskTokenizer,
    'MentionPlusContext':MentionPlusContext,
    'NNDescentSearch':NNDescentSearch,
    'MeanPooling': MeanPooling,
    'CLSMeanPooling': CLSMeanPooling,
    'PadMention': PadMention,
    'MentionPooling': MentionPooling,
    'BiEncoderDataset': BiEncoderDataset,
    'CrossEncoderTokenizer':CrossEncoderTokenizer,
    'CrossEncoderDataset': CrossEncoderDataset,
    'BinaryCrossEntropyLoss':BinaryCrossEntropyLoss
}