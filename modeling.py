from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import numpy as np
from zeshel import get_corpus_names
from zeshel import get_doc_similarities
from tqdm.auto import tqdm

def param_search(X, y, model, param_grid, n_split=5, n_iter=100, random_state=0, n_jobs=10, verbose=0):
    cross_val = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    grid_search = RandomizedSearchCV(model, param_grid, scoring='roc_auc', n_iter=n_iter, verbose=verbose, random_state=random_state,
                            n_jobs=n_jobs, refit=True, cv=cross_val)

    # Fit GridSearchCV
    grid_search.fit(X, y)

    return grid_search


def theshold_tuning(X, y, model, cv=True, n_splits=5, start=0, inc=0.01, random_state=0):
    
    if cv:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        y_val_proba = []
        y_val_true = []

        for train_index, val_index in kfold.split(X, y):   
            X_training, X_val = X[train_index], X[val_index]
            y_training, y_val = y[train_index], y[val_index]
            y_val_true.append(y_val)
            model.fit(X_training, y_training)
            y_val_proba.append(model.predict_proba(X_val))

        y_val_proba = np.concatenate(y_val_proba)
        y_val_true = np.concatenate(y_val_true)
    else:
        y_val_proba = model.predict_proba(X)
        y_val_true = y

    thresholds = np.arange(start, 1, inc)
    f1_scores = []

    for t in tqdm(thresholds):
        y_pred = (y_val_proba[:, 1] >= t).astype(int)
        f1_scores.append(f1_score(y_val_true, y_pred, average='macro'))

    return thresholds[np.argmax(f1_scores)], np.sort(f1_scores)


def evaluate_clf(split, annset, model, threshold, testing=True, get_similarities=True, X=None, y=None):

        y_proba = []
        ys = []

        if get_similarities:
            themes = list(get_corpus_names(split)[1].keys())
            for theme in themes:
                    theme_docs = [doc for doc in split if doc.features['corpus'] == theme]

                    anns = []
                    gdoc_ids = []
                    for i, gdoc in enumerate(theme_docs):
                            for ann in gdoc.annset(annset):
                                    anns.append(ann)
                                    gdoc_ids.append(i)
                    
                    X, y, _ = get_doc_similarities(anns, testing=testing)
                    y_proba.append(model.predict_proba(X))
                    ys.append(y)

            y_proba = np.concatenate(y_proba)
            y = np.concatenate(ys)

        else:
            y_proba = model.predict_proba(X)
            ys = y
                
        y_pred = (y_proba[:, 1] >= threshold).astype(int)                             
   
        np.save('predictions', np.c_[y, y_pred])
        
        print(classification_report(y, y_pred, digits=4))