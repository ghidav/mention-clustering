import yaml
from module.datamodule import CrossEncoderDataModule
from lightning import Trainer, seed_everything
from module.crossencoder import CrossEncoder
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

with open('/home/dghilardi/mention_clustering/mention-clustering/model_scripts/config_crossencoder/config_train.yaml', 'r') as f:
    config = yaml.safe_load(f)

seed_everything(config['seed'])

crossencoder_data = CrossEncoderDataModule(config['data'], config['tokenizer'])
crossencoder_model = CrossEncoder(config['encoder'], config['optimizer'], config['loss'])

if config['wandb']['enable']:
    logger = WandbLogger(**config['wandb']['params'])
else:
    logger = False

callbacks = []
if config['early_stopping']['enable']:
    early_stopping = EarlyStopping(**config['early_stopping']['params'])
    callbacks.append(early_stopping)

if config['model_checkpoint']['enable']:
    model_checkpoint = ModelCheckpoint(**config['model_checkpoint']['params'])
    callbacks.append(model_checkpoint)

crossencoder_data.prepare_data()
# trainer = Trainer(**config['trainer'], logger=logger, callbacks=callbacks)
# trainer.fit(crossencoder_model, crossencoder_data)
# trainer.validate(crossencoder_model, crossencoder_data, ckpt_path=model_checkpoint.best_model_path)
# trainer.test(crossencoder_model, crossencoder_data, ckpt_path=model_checkpoint.best_model_path)