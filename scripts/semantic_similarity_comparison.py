from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.spatial.distance import minkowski
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from networkx.algorithms.traversal.depth_first_search import dfs_tree


def get_sim(term1, term2=None):
    if term2 == None:
        assert type(term1) == tuple or type(term1) == list
        temp = term1
        term1, term2 = temp
    if "UBERON" in term1 and "UBERON" in term2:
        t1 = set(subsumers.get(term1, [term1]))
        t2 = set(subsumers.get(term2, [term2]))
        if len(set.union(t1, t2)) > 0:
            simj=len(set.intersection(t1, t2))/len(set.union(t1, t2))
        else:
            simj = 0.0
    else:
        simj = 0.0
    return simj

def score_by_metric(X, Y, metric: str):
    if metric == "Jaccard similarity":
        score = get_sim(X, Y)
    else:
        x_emb, y_emb = embeddings.get(X), embeddings.get(Y)
        if metric == "Cosine similarity":
            score = norm_embeddings.get(X).dot(norm_embeddings.get(Y))
        elif metric == "Dot product":
            score = x_emb.dot(y_emb)
        if metric == "Minkowski distance":
            score = minkowski(x_emb, y_emb, p=3)
        if metric == "Euclidean distance":
            score = minkowski(x_emb, y_emb, p=2)
        if metric == "Manhattan distance":
            score = minkowski(x_emb, y_emb, p=1)
    return score

BASE_DIR = Path(".").absolute()
DATA_DIR = Path.joinpath(BASE_DIR, "data")
ANNOT_DIR = Path.joinpath(DATA_DIR, "annotations", "UBERON")
GRAPH_FILE = Path.joinpath(DATA_DIR, "uberon_hierarchy.graphml")
EMB_FILE = Path.joinpath(DATA_DIR, "embeddings", "140_25_1.521_0.751_NODE2VEC_UBERON_OBO_128D.npy")
MERGED_DATASET = Path.joinpath(ANNOT_DIR, "merged_dataset.csv")


if __name__ == "__main__":
    if not EMB_FILE.exists():
        print(f"\nEmbedding file not found in \n\t:\"{EMB_FILE}\".\nMake sure that the embedding file is named correctly.")
        print("If you do not have the .npy file, run `NODE2VEC_on_UBERON.ipynb`.")
    else:
        print(f"\nEmbedding file found in \"{EMB_FILE}\".\n\tLoading embeddings ...")
        embeddings = np.load(EMB_FILE, allow_pickle=True)
        emb_dim = embeddings[0][1].ndim
        embeddings = dict((k, v.flatten() if emb_dim == 2 else v) for k, v in embeddings)
        print(f"\tNormalizing embeddings ...")
        norm_embeddings = dict((k, v/np.linalg.norm(v)) for k, v in embeddings.items())
        for k, v in embeddings.items(): break
        print(f"\t# embeddings: {len(norm_embeddings)}\n\tEmbedding dimension: {v.shape[-1]}")

    if not GRAPH_FILE.exists():
        print(f"\n{GRAPH_FILE} does not exist.\nPlease try running 'dataset_generator.py' from './scripts'.")
    else:
        print(f"\nUBERON digraph already exists\n\tat: \"{GRAPH_FILE}\".\n\tLoading digraph ...")
        uberon_digraph = nx.read_graphml(GRAPH_FILE)
        print(f"\t\t# nodes: {uberon_digraph.number_of_nodes()}\n"\
                f"\t\t# edges: {uberon_digraph.number_of_edges()}")

    subsumers = dict((i,list(
        set(np.array(dfs_tree(uberon_digraph, i).edges()).flatten().tolist() + [i]) - 
        set(["root"]))) for i in uberon_digraph.nodes())
    
    pd_merged = pd.read_csv(MERGED_DATASET)
    pd_merged["combination"] = pd_merged["combination"].apply(literal_eval)
    data = [j for i in pd_merged["combination"].values.tolist() for j in i]

    metrics = [
        'Dot product', 'Cosine similarity', 'Euclidean distance',
        'Manhattan distance', 'Minkowski distance', 'Jaccard similarity',  
    ]

    all_scores = []
    pbar = tqdm(total=len(metrics), desc="Calculating accuracy of RF")
    for metric in metrics:
        pbar.set_description(f"Calculating accuracy of RF for {metric}")
        dataset = []
        for i in data:
            temp = []
            for j in i[:-1]:
                temp.extend([j, score_by_metric(j, i[-1], metric)])
            temp.append(i[-1])
            dataset.append(temp)
        dataset = pd.DataFrame(
            dataset, columns=[
                'term_x', '{0}_x'.format(metric), 'term_y', '{0}_y'.format(metric),
                'term_z', '{0}_z'.format(metric), 'term_alpha'])
        dataset.drop_duplicates(inplace=True)
        all_terms = dataset.filter(regex='^term', axis=1)
        tags = sorted(np.unique(all_terms.values.flatten()))
        tag_to_idx = dict((i, idx) for idx, i in enumerate(tags))
        idx_to_tag = dict((v, k) for k, v in tag_to_idx.items())

        dataset['term_x'] = dataset['term_x'].apply(lambda x: tag_to_idx.get(x))
        dataset['term_y'] = dataset['term_y'].apply(lambda x: tag_to_idx.get(x))
        dataset['term_z'] = dataset['term_z'].apply(lambda x: tag_to_idx.get(x))
        dataset['term_alpha'] = dataset['term_alpha'].apply(lambda x: tag_to_idx.get(x))
        
        X = dataset.iloc[:,:-1].values
        Y = dataset['term_alpha'].values
        
        kfold = KFold(n_splits=10, shuffle=True, random_state=2023)

        results = []
        clf = RandomForestClassifier(n_estimators=500, criterion='entropy')
        for idx, (train_index, test_index) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            clf.fit(X_train, Y_train)
            Y_hat = clf.predict(X_test)
            results.append([Y_test, Y_hat])
        all_scores.append([
            metric,
            np.round(np.mean([accuracy_score(i[0], i[1]) for i in results]), 2),
            np.round(np.mean([f1_score(i[0], i[1], average="weighted") for i in results]), 2)
        ])
        pbar.update(1)

    print("\n\n10 Fold Cross Validation by metrics:\n")
    max_len = max([len(i) for i in metrics]) + 6
    for idx, metric in enumerate([["Metric", "Mean accuracy","Mean F1 score"]] + all_scores):
        start_ws = " "*int((max_len-len(metric[0]))*0.5)
        res_str = f"{start_ws}{metric[0]}{' '*(max_len-len(start_ws)-len(metric[0]))}|"
        len_str = "-"*(len(res_str)-1) + "+"
        acc, f1 = str(metric[1]), str(metric[2])
        acc_ws = " "*int((17-len(acc))*0.5)
        acc_str = f"{acc_ws}{acc}{' '*(17-len(acc_ws)-len(acc))}|"
        len_str += "-"*(len(acc_str)-1) + "+"
        f1_ws = " "*int((17-len(f1))*0.5)
        f1_str = f"{f1_ws}{f1}{' '*(17-len(f1_ws)-len(f1))}"
        len_str += "-"*len(f1_str)
        len_str = f"\n{len_str}" if idx != len(all_scores) else "\n"
        res_str += acc_str + f1_str
        print(res_str + len_str)