import jsonlines
from gatenlp import Document
import numpy as np
import lightning as L


class ZeshelPreprocessing(object):

    def __init__(self) -> None:
        pass

    def get_corpus_names(self, gdocs):
        names = [gdoc.features['corpus'] for gdoc in gdocs]
        mapping = {key: value for value, key in enumerate(np.unique(names))}
    
        return np.array([mapping[item] for item in names]), mapping

    def load_gdocs(self, name, path=''):
        gdocs = []
        # Load documents from a JSON Lines file
        with jsonlines.open(path+f"datasets/{name}.jsonl", 'r') as f:
            for line in f.iter():
                gdocs.append(Document().from_dict(line))
        return gdocs
    
    def get_examples(self, name, path='', annset='Gold'):
        gdocs = self.load_gdocs(name, path)
        corpus, mapping = self.get_corpus_names(gdocs)
        examples = []
        for theme in mapping.keys():
            mask = corpus == mapping[theme]
            filtered_gdocs = [gdoc for gdoc, is_selected in zip(gdocs, mask) if is_selected]
            examples += self.get_corpus_examples(filtered_gdocs, theme, annset)
        return examples

    def get_corpus_examples(self, gdocs, theme, annset='Gold'):
            examples = []
            for gdoc in gdocs:
                for ann in gdoc.annset(annset):
                    mention = gdoc.text[ann.start:ann.end]
                    left_context = gdoc[:ann.start].replace('\"','')
                    right_context = gdoc[ann.end:].replace('\"','')
                    examples.append({'left_context': left_context, 'mention': mention, 'right_context':right_context,
                          'corpus':theme, 'entity_id':ann.features['entity_id']})
            return examples

