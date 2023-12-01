import yaml
import pickle
from pathlib import Path
from module.datamodule import BiEncoderDataModule
from module.biencoder import BiEncoderNoMention
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

with open('/home/dghilardi/mention_clustering/mention-clustering/model_scripts/config/config_predict.yaml', 'r') as f:
    config = yaml.safe_load(f)

seed_everything(config['seed'])
biencoder_data = BiEncoderDataModule(config['data'], config['tokenizer'], config['mention_encoder'], config['knn_search'])
biencoder_model = BiEncoderNoMention(config['context_encoder'], config['optimizer'], config['loss'], config['miner'], config['knn_search'])

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

trainer = Trainer(**config['trainer'], logger=logger, callbacks=callbacks)
trainer.predict(biencoder_model, biencoder_data, ckpt_path=config['model_ckpt'])
x=0

output_folder = Path(config['data']['output_biencoder_folder'])
output_folder.mkdir(exist_ok=True, parents=True)
with open(output_folder / config['data']['output_biencoder_filename'], 'wb') as f:
    pickle.dump((biencoder_model.pred_dict_output, biencoder_model.pred_pairs), f, protocol=pickle.HIGHEST_PROTOCOL)
