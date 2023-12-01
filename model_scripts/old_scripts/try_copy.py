import yaml
from module.datamodule import BiEncoderDataModule
from module.biencoder import BiEncoder
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

with open('/home/dghilardi/mention_clustering/mention-clustering/pytorch_model/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

seed_everything(config['seed'])
biencoder_data = BiEncoderDataModule(config['data'], config['tokenizer'], config['mention_encoder'], config['knn_search'])
biencoder_model = BiEncoder(config['context_encoder'], config['optimizer'], config['loss'], config['miner'])

if config['wandb']['enable']:
    logger = WandbLogger(**config['wandb']['params'])
else:
    logger = False

callbacks = []
if config['early_stopping']['enable']:
    early_stopping = callbacks.append(EarlyStopping(**config['early_stopping']['params']))
if config['model_checkpoint']['enable']:
    model_checkpoint = callbacks.append(ModelCheckpoint(**config['model_checkpoint']['params']))

trainer = Trainer(**config['trainer'], logger=logger, callbacks=callbacks)
trainer.fit(biencoder_model, biencoder_data)
x=0
