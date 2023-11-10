import numpy as np
import itertools
from tqdm.auto import tqdm
from utils import vector_decode, sourface_similarity, cosine_similarity

def get_corpus_names(gdocs):
    names = [gdoc.features['corpus'] for gdoc in gdocs]
    mapping = {key: value for value, key in enumerate(np.unique(names))}
    
    return np.array([mapping[item] for item in names]), mapping

def get_doc_similarities(anns, factor=1, testing=False):
    matr_embd = np.array([vector_decode(ann.features['encodings']) for ann in anns])

    indexes = []
    target = []
    similarities = []

    # Compute the number of matches
    k = 0
    m = 0
    n = len(anns)
    for (idx_a, idx_b) in tqdm(itertools.combinations(range(n), 2), total=n*(n-1)/2):
        eid_a = anns[idx_a].features['entity_id']
        eid_b = anns[idx_b].features['entity_id']
        if eid_a and eid_b:
            if (eid_a == eid_b) or (k < m*factor) or testing:
                
                if eid_a == eid_b:
                    m += 1
                else:
                    k += 1

                target.append(int(eid_a == eid_b))

                text_a = anns[idx_a].features['mention']
                text_b = anns[idx_b].features['mention']

                sur_sim = sourface_similarity(text_a, text_b)
                
                emb_a = matr_embd[idx_a]
                emb_b = matr_embd[idx_b]
                emb_sim = cosine_similarity(emb_a, emb_b)

                similarities.append(sur_sim + [emb_sim])
                indexes.append([idx_a, idx_b])

    return np.array(similarities), np.array(target), np.array(indexes)

def get_zeshel_similarities(gdocs, annset, factor=1, testing=False):
    
    corpus, mapping = get_corpus_names(gdocs)
    dfs = {}
    
    for theme in mapping.keys():
        mask = corpus == mapping[theme]
        filtered_docs = [gdoc for gdoc, is_selected in zip(gdocs, mask) if is_selected]

        all_anns = []

        for gdoc in filtered_docs:
            for ann in gdoc.annset(annset):
                all_anns.append(ann)
    
        dfs[theme] = get_doc_similarities(all_anns, factor=factor, testing=testing)

    return dfs