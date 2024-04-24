import os
import sys
import random
from pathlib import Path
from itertools import combinations
from argparse import ArgumentParser

import fastobo
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import urllib.request as request
from tqdm.auto import tqdm
from hyperopt import tpe, hp, fmin, Trials
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.traversal.depth_first_search import dfs_tree

random.seed(2023)

# Defining configuration to train an embedding model from ontology hierarchy
parser = ArgumentParser(description='Graph Embedding on UBERON Ontology DiGraph')
parser.add_argument('-d', '--dimension',
    help='Embedding Dimension (default: 128)', default=128, type=int)
parser.add_argument('-e', '--epochs',
    help='number of iterations for embedding training (default: 2)',
    default=2, type=int)
parser.add_argument('-b', '--batch-size',
    help='Batch size for training (default: 50)',
    default=50, type=int)
parser.add_argument('-f', '--savefile',
    help='whether or not to save the embeddings (default: False)',
    default=False, type=bool)
parser.add_argument('-wn', '--walk_number',
    help='number of random walks from a source node',
    default=150, type=int, choices=list(range(50, 160, 10)))
parser.add_argument('-wl', '--walk_length',
    help='number of hops from the source node',
    default=30, type=int, choices=list(range(5, 35, 5)))
parser.add_argument('-p', '--p',
    help='return parameter, higher p - less likely to return to already visited node',
    default=1.0, type=float, choices=[round(i*0.1, 1) for i in range(10, 21)])
parser.add_argument('-q', '--q',
    help='in-out parameter, q < 1 - walk more inclined to visit nodes further away',
    default=0.1, type=float, choices=[round(i*0.1, 1) for i in range(1, 10)])

config = vars(parser.parse_args())
print(f"Python command: 'python {' '.join(sys.argv)}'")


BASE_DIR = Path(".").absolute().parent
DATA_DIR = Path.joinpath(BASE_DIR, "data")
ANNOT_DIR = Path.joinpath(DATA_DIR, "annotations", "UBERON")
GRAPH_FILE = Path.joinpath(DATA_DIR, "uberon_hierarchy.graphml")
EMB_DIR = Path.joinpath(DATA_DIR, "grid_search", "embeddings")
SAMPLE_FILE = Path.joinpath(DATA_DIR, "samples.csv")
EMB_DIR.mkdir(exist_ok=True, parents=True)

devices = tf.config.list_physical_devices("GPU")
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk, UnsupervisedSampler
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification


if not GRAPH_FILE.exists():
    uberon_subset = "basic"
    uberon_url = f"http://purl.obolibrary.org/obo/uberon/{uberon_subset}.obo"
    print(f"Downloading {uberon_subset}.obo from {uberon_url} ...")

    with request.urlopen(uberon_url) as response:
        pato = fastobo.load(response)

    print(f"Creating networkx digraph from UBERON nodes and edges ...")
    uberon_digraph = nx.DiGraph()
    for frame in pato:
        if isinstance(frame, fastobo.term.TermFrame):
            uberon_digraph.add_node(str(frame.id))
            for clause in frame:
                if isinstance(clause, fastobo.term.IsAClause):
                    uberon_digraph.add_edge(str(frame.id), str(clause.term))
    print(f"Saving networkx graph in graphml format ...")
    nx.write_graphml(uberon_digraph, GRAPH_FILE)
    print(f"Graph saved to \"{GRAPH_FILE}\".\nYou can load the graph later using: nx.read_graphml(\"{GRAPH_FILE}\")")
else:
    print(f"UBERON digraph already exists at: \"{GRAPH_FILE}\".\nLoading digraph ...")
    uberon_digraph = nx.read_graphml(GRAPH_FILE)
print(f"# nodes: {uberon_digraph.number_of_nodes()}\n"\
        f"# edges: {uberon_digraph.number_of_edges()}")

subsumers = dict((i,list(
    set(np.array(dfs_tree(uberon_digraph, i).edges()).flatten().tolist() + [i]) - 
    set(["root"]))) for i in uberon_digraph.nodes())


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


if __name__ == "__main__":
    # We train the model using Node2Vec algorithm - using Biased Random Walk. Once trained,
    # we test and compare different model's (models trained with different hyperparameters)
    # understanding on semantic relationship between different concepts.
    # 1. From all possible combination of a pair of concepts - e.g. (UBERON:0000000, UBERON:0000002),
    # we create two groups, A. where the pair has semantic relationship (semantic similarity calculated
    # using jaccard) - called child-parent nodes and B. where the pair has no semantic relationship - called
    # random nodes.
    # 2. We take 1% of total combination as sample population out of which 50% are random samples from
    # child-parent nodes and 50% are from random nodes.
    # 3. Once a model is trained, we compute the cosine similarity between all pairs from both groups A and B.
    # 4. The mean similarity from each group is compared with their respective ground truth mean (mean of 
    # jaccard semantic similarity)
    # Using the scoring method defined below: the score can be between 1 and 2, 1 denoting poor performance and 2
    # denoting best performance.

    if not SAMPLE_FILE.exists(): # If SAMPLE_FILE is not already created, create and save sample using step 1 and 2
                                    # described above.
        print(f"Creating combination of 2 nodes from all UBERON nodes ...")
        all_nodes = list(uberon_digraph.nodes())
        uberon_comb = list(combinations(all_nodes, 2))

        sample_size = int(0.01*len(uberon_comb))
        print(f"Separating combinations of child-parent nodes and random nodes ...")

        cp_nodes, rand_nodes = [], []
        for comb in tqdm(uberon_comb):
            jaccard_sim = get_sim(*comb)
            if jaccard_sim > 0.4:
                cp_nodes.append([comb[0], comb[1], jaccard_sim])
            else:
                if jaccard_sim == 0:
                    rand_nodes.append([comb[0], comb[1], jaccard_sim])

        print(f"# Child-parent nodes combination with jaccard semantic similarity > 0.4: {len(cp_nodes)}")
        print(f"# Random nodes combination: {len(rand_nodes)}")

        print(f"Selecting 0.05% of samples from child-parent combination and 0.05% from random nodes combination ...")
        cp_samples = random.choices(cp_nodes, k=sample_size//2)
        rand_samples = random.choices(rand_nodes, k=sample_size-len(cp_samples))
        print(f"# sample of child-parent nodes: {len(cp_samples)}")
        print(f"# sample of random nodes: {len(rand_samples)}")

        print(f"Saving sample info to {SAMPLE_FILE} ...")
        pd_data = pd.DataFrame(cp_samples+rand_samples, columns=["node_1", "node_2", "jaccard_sim"])
        pd_data["type"] = ["child-parent"]*len(cp_samples)+["random"]*len(rand_samples)
        pd_data.to_csv(SAMPLE_FILE, index=False)
        print(f"Samples saved. You can load the samples from {SAMPLE_FILE} as pandas dataframe.")
    else:
        print(f"Random sample data already exists. Loading file: {SAMPLE_FILE} ...")
        pd_data = pd.read_csv(SAMPLE_FILE)
        cp_samples = pd_data[pd_data["type"]=="child-parent"].iloc[:,:-1].values.tolist()
        rand_samples = pd_data[pd_data["type"]=="random"].iloc[:,:-1].values.tolist()
        print(f"# sample of child-parent nodes: {len(cp_samples)}")
        print(f"# sample of random nodes: {len(rand_samples)}")
    
    # Training using StellarGraph node2vec algorithm
    G = StellarGraph.from_networkx(uberon_digraph)
    print(G.info())

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        walker = BiasedRandomWalk(
            G,
            n=config.get("walk_number"),
            length=config.get("walk_length"),
            p=config.get("p"),
            q=config.get("q"),
        )
        unsupervised_samples = UnsupervisedSampler(G, nodes=list(G.nodes()), walker=walker)
        generator = Node2VecLinkGenerator(G, config.get("batch_size"))
        node2vec = Node2Vec(config.get("dimension"), generator=generator)
        x_inp, x_out = node2vec.in_out_tensors()
        prediction = link_classification(
            output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
        )(x_out)

        model = tf.keras.Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3 ),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[tf.keras.metrics.binary_accuracy],
        )
    print("Training model ...")
    history = model.fit(
        generator.flow(unsupervised_samples),
        epochs=config.get("epochs"),
        verbose=True,
        use_multiprocessing=False,
        shuffle=True,
    )
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = tf.keras.Model(inputs=x_inp_src, outputs=x_out_src)

    node_gen = Node2VecNodeGenerator(G, config.get("batch_size")).flow(G.nodes())
    node_embeddings = embedding_model.predict(node_gen, workers=os.cpu_count(), verbose=1)

    print("Generating embeddings ...")
    embeddings = np.vstack([np.array((i, node_embeddings[idx].reshape(1, -1)
                                      ), dtype=object) for idx, i in enumerate(G.nodes())])
    emb_dict = dict((k, v) for k, v in embeddings)
    
    print("Computing scores ...")
    cosine_sim = [cosine_similarity(emb_dict[i[0]], emb_dict[i[1]])[0][0] for i in cp_samples]
    rand_sim = [cosine_similarity(emb_dict[i[0]], emb_dict[i[1]])[0][0] for i in rand_samples]
    cp_score = 1 / (1 + abs(0.5 - np.mean(cosine_sim)))
    rand_score = 1 / (1 + abs(0.- - np.mean(rand_sim)))
    final_score = np.mean(cp_score + rand_score)
    print(f"Final score: {final_score}")

    if config.get("savefile"):
        with open(Path.joinpath(EMB_DIR, f"{str(round(final_score, 6))}_" + \
            "_".join([str(i) for i in config.values()][3:]) + "_" + \
            f"NODE2VEC_UBERON_OBO_{config.get('dimension')}D.npy"), "wb") as f:
            np.save(f, embeddings)
