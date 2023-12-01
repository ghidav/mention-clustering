from pytorch_metric_learning.losses import MultiSimilarityLoss

class MSLoss:

    def __init__(self, loss_params):
        self.loss = MultiSimilarityLoss(**loss_params)
    
    def compute_loss(self, input_for_loss):
        return self.loss(*input_for_loss)