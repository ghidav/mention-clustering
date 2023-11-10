from gatenlp import Document
from pathlib import Path
import json
import numpy as np
import requests
import itertools
import base64
import textdistance
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import KFold
import networkx as nx
from networkx.algorithms.community import louvain_communities
import requests
import numpy as np
import base64
from requests.auth import HTTPBasicAuth
from tqdm.auto import tqdm
import jsonlines

similarity_name = np.array([
    'jaro',
    'jaro_bi',
    'jaro_tri',
    'jaro_four',
    'ham',
    'ham_bi',
    'ham_tri',
    'ham_four',
    'damer',
    'damer_bi',
    'damer_tri',
    'damer_four',
    'jaccard',
    'cosine',
    'bag',
    'substring',
    'subsequence',
    'identity',
    'embedding'])

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def save_gdocs(gdocs, name):
    jsonl_file_path = f"datasets/{name}.jsonl"

    with open(jsonl_file_path, 'w') as file:
        for gdoc in tqdm(gdocs):

            doc_json = json.dumps(gdoc.to_dict())
            
            # Write the JSON string to the file followed by a newline
            file.write(doc_json + '\n')

def load_gdocs(name):
    gdocs = []
    # Load documents from a JSON Lines file
    with jsonlines.open(f"datasets/{name}.jsonl", 'r') as f:
        for line in tqdm(f.iter()):
            gdocs.append(Document().from_dict(line))
    
    return gdocs

def get_mention_encodings(mention, ctx_left, ctx_right):

    res = requests.post('http://localhost:20980/api/blink/biencoder/mention',
                        auth = HTTPBasicAuth('DS2023', 'eexeegheichai3OhChi5AhcheecaaShe'),
                        json = [{'context_left': ctx_left, 'mention': mention, 'context_right': ctx_right}],
                        verify=False)

    return res.json()['encodings'][0]

def encode_mentions(gdoc, annset_name):
    annset = gdoc.annset(annset_name)

    text = gdoc.text

    for ann in annset:
        start = ann.start
        end = ann.end
        
        encodings = get_mention_encodings(text[start:end], text[:start], text[end:])

        ann.features['encodings'] = encodings

def encode_docs(gdocs, annset_name):
    for gdoc in tqdm(gdocs):
        encode_mentions(gdoc, annset_name)

# ---------------------------------------------------------------

def import_document(file_path, get_embd=False, api=''):
    # import
    with open(file_path, 'r') as f:
        doc = json.load(f)
    # get embedding
    if get_embd:
        doc = requests.post(api, json=doc)
        doc = Document.from_dict(doc.json())
    else:
        doc = Document.from_dict(doc)
    return doc

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b))

def sourface_similarity(text_a, text_b):
    jaro = textdistance.JaroWinkler()
    jaro_bi = textdistance.JaroWinkler(qval=2)
    jaro_tri = textdistance.JaroWinkler(qval=3)
    jaro_four = textdistance.JaroWinkler(qval=4)
    ham = textdistance.Hamming()
    ham_bi = textdistance.Hamming(qval=2)
    ham_tri = textdistance.Hamming(qval=3)
    ham_four = textdistance.Hamming(qval=4)
    damer = textdistance.DamerauLevenshtein()
    damer_bi = textdistance.DamerauLevenshtein(qval=2)
    damer_tri = textdistance.DamerauLevenshtein(qval=3)
    damer_four = textdistance.DamerauLevenshtein(qval=4)
    jaccard = textdistance.Jaccard()
    cosine = textdistance.Cosine()
    bag = textdistance.Bag()
    substring = textdistance.LCSStr()
    subsequence = textdistance.LCSSeq()
    identity = textdistance.Identity()

    score = [jaro.normalized_similarity(text_a, text_b),
             jaro_bi.normalized_similarity(text_a, text_b),
             jaro_tri.normalized_similarity(text_a, text_b),
             jaro_four.normalized_similarity(text_a, text_b),
             ham.normalized_similarity(text_a, text_b),
             ham_bi.normalized_similarity(text_a, text_b),
             ham_tri.normalized_similarity(text_a, text_b),
             ham_four.normalized_similarity(text_a, text_b),
             damer.normalized_similarity(text_a, text_b),
             damer_bi.normalized_similarity(text_a, text_b),
             damer_tri.normalized_similarity(text_a, text_b),
             damer_four.normalized_similarity(text_a, text_b),
             jaccard.normalized_similarity(text_a, text_b),
             cosine.normalized_similarity(text_a, text_b),
             bag.normalized_similarity(text_a, text_b),
             substring.normalized_similarity(text_a, text_b),
             subsequence.normalized_similarity(text_a, text_b),
             identity.normalized_similarity(text_a, text_b)]
    return score

def sourface_similarity_light(text_a, text_b):
    jaro = textdistance.JaroWinkler()
    jaro_bi = textdistance.JaroWinkler(qval=2)
    jaro_tri = textdistance.JaroWinkler(qval=3)
    jaro_four = textdistance.JaroWinkler(qval=4)
    damer_bi = textdistance.DamerauLevenshtein(qval=2)
    jaccard = textdistance.Jaccard()
    cosine = textdistance.Cosine()
    bag = textdistance.Bag()
    subsequence = textdistance.LCSSeq()

    score = [jaro.normalized_similarity(text_a, text_b),
             jaro_bi.normalized_similarity(text_a, text_b),
             jaro_tri.normalized_similarity(text_a, text_b),
             jaro_four.normalized_similarity(text_a, text_b),
             damer_bi.normalized_similarity(text_a, text_b),
             jaccard.normalized_similarity(text_a, text_b),
             cosine.normalized_similarity(text_a, text_b),
             bag.normalized_similarity(text_a, text_b),
             subsequence.normalized_similarity(text_a, text_b)]
    return score

def compute_similarity(idx_a, idx_b, annset, matr_embd, decimal='', uncased=True):
    text_a = annset[idx_a].features['mention']
    text_b = annset[idx_b].features['mention']
    if uncased:
        text_a = text_a.lower()
        text_b = text_b.lower()
    embd_a = matr_embd[idx_a]
    embd_b = matr_embd[idx_b]
    score = sourface_similarity(text_a, text_b) + [cosine_similarity(embd_a, embd_b)]
    if decimal:
        score = [round(s, decimal) for s in score]
    return score

def compute_similarity_light(idx_a, idx_b, annset, matr_embd, decimal='', uncased=True):
    text_a = annset[idx_a].features['mention']
    text_b = annset[idx_b].features['mention']
    if uncased:
        text_a = text_a.lower()
        text_b = text_b.lower()
    embd_a = matr_embd[idx_a]
    embd_b = matr_embd[idx_b]
    score = sourface_similarity_light(text_a, text_b) + [cosine_similarity(embd_a, embd_b)]
    if decimal:
        score = [round(s, decimal) for s in score]
    return score

# CROSS VALIDATION FUNCTION
def cv_loop(X, y, model, n_split, seed):
    kfold = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    y_val_pred = []
    y_val_logits = []
    y_val_true = []

    for train_index, val_index in kfold.split(X):   
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        y_val_true.append(y_val)
        model.fit(X_train, y_train)
        y_val_pred.append(model.predict(X_val))
        y_val_logits.append(model.predict_proba(X_val))
    y_val_pred = np.concatenate(y_val_pred)
    y_val_logits = np.concatenate(y_val_logits)
    y_val_true = np.concatenate(y_val_true)
    fpr, tpr, _ = metrics.roc_curve(y_val_true, y_val_logits[:,1], pos_label=1)

    print('CROSS VALIDATION RESULTS')
    print(f'AUC:{metrics.auc(fpr, tpr)}')
    print(classification_report(y_val_true, y_val_pred))

def make_clusters(doc, annset_name, model, threshold, seed=11, separate_annset=True):
    annset = doc.annset(annset_name)
    matr_embd = np.array([vector_decode(ann.features['linking']['encoding']) for ann in annset])
    pairs = np.array(list(itertools.combinations(range(len(annset)), 2)))
    # compute similarity
    X_test=[]
    for idx_a, idx_b in pairs:
        X_test.append(compute_similarity_light(idx_a, idx_b, annset, matr_embd, 5, uncased=True))
    X_test = np.array(X_test)
    # compute probabilities
    proba = model.predict_proba(X_test)
    # make graph
    graph = nx.Graph()
    similarity_matrix = np.identity(len(annset))
    for p, (idx_a, idx_b) in zip(proba[:,1], pairs):
        similarity_matrix[idx_a, idx_b] = p
        similarity_matrix[idx_b, idx_a] = p
        if p>=threshold:
            graph.add_edge(idx_a, idx_b, weight=p)
    # create clusters
    clusters = louvain_communities(graph, resolution=2, seed=seed)
    clusters = [list(c) for c in clusters]
    # refine cluster
    clusters_refine = []
    for cluster in clusters:
        new_cluster = []
        for idx_target in cluster:
            score = []
            for idx_other in list(set(cluster) - set([idx_target])):
                score.append(similarity_matrix[idx_target, idx_other])
            mean_score = np.array(score).mean()
            if mean_score>=threshold:
                new_cluster.append(idx_target)
        clusters_refine.append(new_cluster)
    # create cluster2id
    cluster2id = {}
    ann_in_cluster = []
    for c_idx, cluster in enumerate(clusters_refine):
        cluster2id[c_idx] = cluster
        ann_in_cluster += cluster
    clust_id = max(cluster2id.keys()) + 1
    for ann_id in range(len(annset)):
        if ann_id not in ann_in_cluster:
            cluster2id[clust_id] = [ann_id]
            clust_id += 1
    # create id2cluster
    id2cluster = {}
    for key, values in cluster2id.items():
        for idx in values:
            id2cluster[idx] = key
    # create annotation set
    # build annotation set
    if separate_annset:
        try:
            doc.remove_annset('enitites_clustered')
        except:
            pass
        newset = doc.annset('enitites_clustered')
        for ann_id, ann in enumerate(annset):
            newset.add(ann.start, ann.end, f'CLUST-{id2cluster[ann_id]}')
    else:
        for ann_id, ann in enumerate(annset):
            annset[ann_id].features['cluster'] = id2cluster[ann_id]
    
    return doc, cluster2id, id2cluster

def predict_parte_controparte(doc, annset_name, model, threshold, id2cluster, cluster2id, feature_name='parte'):
    annset = doc.annset(annset_name)
    feature_str = doc.features[feature_name].lower()
    similarity_score = []
    for ann in annset:
        mention_ann = ann.features['mention']
        similarity_score.append(sourface_similarity_light(feature_str, mention_ann) + [None])
    similarity_score = model.predict_proba(similarity_score)[:,1]
    similarity_score = np.array(similarity_score)
    target_ann = np.argwhere(similarity_score>threshold)
    if target_ann.shape[0] == 0:
        target_ann = np.argwhere(similarity_score == similarity_score.max())
    selected_clusters = []
    for ann_id in target_ann[0]:
        selected_clusters.append(id2cluster[ann_id])
    selected_clusters = set(selected_clusters)
    for clust_id in selected_clusters:
        for ann_id in cluster2id[clust_id]:
            annset[ann_id].features['inferred_type'] = [feature_name]
    return doc, selected_clusters