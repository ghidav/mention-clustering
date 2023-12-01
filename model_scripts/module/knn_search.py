from sklearn.neighbors import KDTree
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pynndescent import NNDescent
from scipy.sparse import coo_matrix
from itertools import combinations

class KDTreeSearch:

    def __init__(self, knn_params) -> None:
        self.knn_params = knn_params

    def fit_sparse(self, mention_embeddings):
        self.index = KDTree(mention_embeddings, **self.knn_params)

    def query(self, x, k):
        return self.index.query(x.to_dense(), k)
    
    def find_hard_pairs(self, mention_embeddings, label, ex_idx, k=10):
        # hard_negative -> most similar negatives
        # hard_positive -> less similar positives
        x = mention_embeddings[ex_idx].reshape(1, -1)
        y = label[ex_idx]
        dist, neighbors_idx = self.query(x, k)
        hard_negatives_idx = [idx for idx in neighbors_idx[0] if label[idx]!=y]
        hard_pos_idx = [idx for idx in neighbors_idx[0] if label[idx]==y and idx!=ex_idx]
        hard_pos_idx.reverse() # first position less similar
        return neighbors_idx, hard_pos_idx, hard_negatives_idx
    
    def get_positive_pairs(self, label):
        ent2ex = defaultdict(list)
        ex2positive = defaultdict(list)
        for ex_idx, ent_id in enumerate(label):
            ent2ex[ent_id].append(ex_idx)
        for ex_idx, ent_id in enumerate(label):
            ex2positive[ex_idx] += list(set(ent2ex[ent_id]) - set([ex_idx]))
        return ent2ex, ex2positive
    
    def construct_npairs(self, mention_embeddings, label, k=10, n_neg=2, n_pos=1):
        idx_list = range(len(label))
        ent2ex, ex2positive = self.get_positive_pairs(label)
        npairs = []
        self.fit_sparse(mention_embeddings)
        for ex_idx, x in tqdm(enumerate(mention_embeddings), desc='Anchor point analyzed', total=len(mention_embeddings)):
            if ex2positive[ex_idx]:
                npair = [ex_idx]
                neighbors_idx, hard_pos_idx, hard_negatives_idx = self.find_hard_pairs(mention_embeddings, label, ex_idx, k)

                if hard_negatives_idx:
                    n_neg_real = min(len(hard_negatives_idx),n_neg)
                    npair += hard_negatives_idx[:n_neg_real]
                    if n_neg_real < n_neg:
                        hard_negatives = False
                        n_random_neg = n_neg - n_neg_real
                    else:
                        hard_negatives = True
                else:
                    n_random_neg = n_neg
                    hard_negatives = False
                
                if hard_pos_idx:
                    not_neighbors_positive = list(set(ex2positive[ex_idx]) - set(hard_pos_idx))
                    np.random.shuffle(not_neighbors_positive)
                    hard_pos_idx = not_neighbors_positive + hard_pos_idx
                    npair += hard_pos_idx[:n_pos]
                else:
                    not_neighbors_positive = ex2positive[ex_idx]
                    np.random.shuffle(not_neighbors_positive)
                    hard_pos_idx = not_neighbors_positive + hard_pos_idx
                    npair += hard_pos_idx[:n_pos]
                
                if not hard_negatives:
                    other_negatives = list(set(idx_list) - (set(npair)+set(ex2positive[ex_idx])))
                    np.random.shuffle(other_negatives)
                    npair += other_negatives[:n_random_neg]

                npairs.append(npair)
        
        # remove duplicate
        npairs = sorted(npairs, key=lambda x: set([x[0],x[1]]))
        prev = None
        npairs_new = []
        for row in npairs_new:
            if set([row[0],row[1]]) == prev:
                pass
            else:
                prev = set([row[0],row[1]])
                npairs_new.append(row)
        
        return npairs


class NNDescentSearch(KDTreeSearch):

    def __init__(self, knn_params) -> None:
        super().__init__(knn_params)

    def torch2scipy(self, torch_sparse):
        coo_data = torch_sparse.coalesce()
        values = coo_data.values().cpu().numpy()
        indices = coo_data.indices().cpu().numpy()
        # create scipy sparse matrix
        scipy_sparse = coo_matrix((values, (indices[0], indices[1])), shape=torch_sparse.shape).tocsr()
        return scipy_sparse

    def fit(self, mention_embeddings):
        print('Fitting Index')
        self.index = NNDescent(mention_embeddings, **self.knn_params) # fit directly with sparse embeddings
        self.index.prepare()
    
    def query(self, x, k):
        return self.index.query(x, k)

    def find_hard_pairs(self, mention_embeddings, label, ex_idx, k=10):
        # hard_negative -> most similar negatives
        # hard_positive -> less similar positives
        x = mention_embeddings[ex_idx]
        y = label[ex_idx]
        neighbors_idx, dist = self.query(x, k) # query directly with sparse embeddings
        hard_negatives_idx = [idx for idx in neighbors_idx[0] if label[idx]!=y]
        hard_pos_idx = [idx for idx in neighbors_idx[0] if label[idx]==y and idx!=ex_idx]
        hard_pos_idx.reverse() # first position less similar
        return neighbors_idx, hard_pos_idx, hard_negatives_idx
    
    def construct_npairs(self, mention_embeddings, label, k=10, n_neg=2, n_pos=1):
        idx_list = range(len(label))
        mention_embeddings = self.torch2scipy(mention_embeddings)
        ent2ex, ex2positive = self.get_positive_pairs(label)
        npairs = []
        self.fit(mention_embeddings)
        for ex_idx, x in tqdm(enumerate(mention_embeddings), desc='Anchor point analyzed', total=mention_embeddings.shape[0]):
            if ex2positive[ex_idx]:
                npair = [ex_idx]
                neighbors_idx, hard_pos_idx, hard_negatives_idx = self.find_hard_pairs(mention_embeddings, label, ex_idx, k)

                if hard_negatives_idx:
                    n_neg_real = min(len(hard_negatives_idx),n_neg)
                    npair += hard_negatives_idx[:n_neg_real] # add most similar negative
                    if n_neg_real < n_neg:
                        hard_negatives = False
                        n_random_neg = n_neg - n_neg_real
                    else:
                        hard_negatives = True
                else:
                    n_random_neg = n_neg
                    hard_negatives = False
                
                if hard_pos_idx:
                    # find positives not in neighbour set (i.e. most dissimilar)
                    not_neighbors_positive = list(set(ex2positive[ex_idx]) - set(hard_pos_idx)) 
                    np.random.shuffle(not_neighbors_positive)
                    hard_pos_idx = not_neighbors_positive + hard_pos_idx
                    npair += hard_pos_idx[:n_pos]
                else:
                    not_neighbors_positive = ex2positive[ex_idx]
                    np.random.shuffle(not_neighbors_positive)
                    hard_pos_idx = not_neighbors_positive + hard_pos_idx
                    npair += hard_pos_idx[:n_pos]
                
                if not hard_negatives:
                    other_negatives = list(set(idx_list) - (set(npair)+set(ex2positive[ex_idx])))
                    np.random.shuffle(other_negatives)
                    npair += other_negatives[:n_random_neg]

                npairs.append(npair)
        
        # remove duplicate
        npairs = sorted(npairs, key=lambda x: set([x[0],x[1]]))
        prev = None
        npairs_new = []
        for row in npairs_new:
            if set([row[0],row[1]]) == prev:
                pass
            else:
                prev = set([row[0],row[1]])
                npairs_new.append(row)
        
        return npairs


