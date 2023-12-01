from pytorch_metric_learning.miners import MultiSimilarityMiner

class MSMiner:

    def __init__(self, miner_params) -> None:
        self.miner = MultiSimilarityMiner(**miner_params)
    
    def get_pairs(self, input_for_miner):
        return self.miner(*input_for_miner)
