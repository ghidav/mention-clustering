from module.preprocessing_module import ZeshelPreprocessing
from pathlib import Path
import json
from tqdm import tqdm

zeshel_preprocessing = ZeshelPreprocessing()

path = "/home/dghilardi/mention_clustering/"
outpath = Path('/home/dghilardi/mention_clustering/mention-clustering/zeshel_processed/')
outpath.mkdir(exist_ok=True)

train_examples = zeshel_preprocessing.get_examples('zeshel_train', path)
val_examples = zeshel_preprocessing.get_examples('zeshel_val', path)
test_examples = zeshel_preprocessing.get_examples('zeshel_test', path)

with open(outpath / 'train.json', 'w') as f:
    for ex in tqdm(train_examples):
        json.dump(ex, f, ensure_ascii=False)
        f.write('\n')
with open(outpath / 'val.json', 'w') as f:
    for ex in tqdm(val_examples):
        json.dump(ex, f, ensure_ascii=False)
        f.write('\n')
with open(outpath / 'test.json', 'w') as f:
    for ex in tqdm(test_examples):
        json.dump(ex, f, ensure_ascii=False)
        f.write('\n')