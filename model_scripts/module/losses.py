from pytorch_metric_learning.losses import MultiSimilarityLoss
from torch.nn import BCEWithLogitsLoss
import torch

class MSLoss:

    def __init__(self, loss_params):
        self.loss = MultiSimilarityLoss(**loss_params)
    
    def compute_loss(self, input_for_loss):
        return self.loss(*input_for_loss)
    
class BinaryCrossEntropyLoss(torch.nn.Module):

    def __init__(self, loss_params) -> None:
        super().__init__()
        pos_weight = torch.tensor([loss_params['pos_weight']])
        self.loss = BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def compute_loss(self, input_for_loss):
        return self.loss(*input_for_loss)