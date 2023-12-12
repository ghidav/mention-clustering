from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from module.config2class import class_dict
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score, BinarySpecificityAtSensitivity, BinarySpecificity


class CrossEncoder(L.LightningModule):

    def __init__(self, encoder_config, optimizer_config, loss_config) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = class_dict[encoder_config['name']](encoder_config['model_ckpt'], encoder_config['params'])
        self.optimizer_config = optimizer_config
        self.loss = class_dict[loss_config['name']](loss_config['params'])
        # self.loss.loss = self.loss.loss.to(self.device)

        self.projector = torch.nn.Linear(768, 1, bias=True)
        self.similarity_logistic = torch.nn.Linear(2, 1, bias=True)

        metrics = MetricCollection([
            BinaryAUROC(), BinaryAccuracy(), BinaryRecall(), BinaryPrecision(), BinaryF1Score(), BinarySpecificity()
        ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
    
    def configure_optimizers(self):
        encoder_param = list(self.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in encoder_param
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in encoder_param
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0}]
            
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=float(self.optimizer_config['learning_rate']))
    
    def forward(self, batch):
        input_ids = batch['input_ids'].to(torch.int32)
        attention_mask = batch['attention_mask'].to(torch.int32)
        token_type_ids = batch['token_type_ids'].to(torch.int32)
        crossencoder_embedding = self.encoder(input_ids, attention_mask, token_type_ids)
        crossencoder_logits = self.projector(crossencoder_embedding)
        crossencoder_similarity = torch.sigmoid(crossencoder_logits).squeeze()
        similarity_tensor = torch.stack((batch['mention_sim'], crossencoder_similarity))
        similarity_logits = self.similarity_logistic(torch.t(similarity_tensor)).squeeze()

        return similarity_logits
    
    def training_step(self, batch, batch_idx):
        similarity_logits = self(batch)
        label_link = batch['link'].float()
        loss = self.loss.compute_loss([similarity_logits, label_link])
        self.train_metrics(similarity_logits, label_link.int())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        similarity_logits = self(batch)
        label_link = batch['link'].float()
        loss = self.loss.compute_loss([similarity_logits, label_link])
        self.val_metrics(similarity_logits, label_link.int())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        similarity_logits = self(batch)
        label_link = batch['link'].float()
        loss = self.loss.compute_loss([similarity_logits, label_link])
        self.test_metrics(similarity_logits, label_link.int())
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False)
        return loss