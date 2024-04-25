import numpy as np
import pandas as pd
from urllib import request
from pathlib import Path
from itertools import product
from collections import defaultdict

import fastobo
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree


BASE_DIR = Path(".").absolute()
DATA_DIR = Path.joinpath(BASE_DIR, "data")
ANNOT_DIR = Path.joinpath(DATA_DIR, "annotations", "UBERON")
GRAPH_FILE = Path.joinpath(DATA_DIR, "uberon_hierarchy.graphml")
EMB_FILE = Path.joinpath(DATA_DIR, "embeddings", "NODE2VEC_UBERON_OBO_128D.npy")
MERGED_DATASET = Path.joinpath(ANNOT_DIR, "merged_dataset.csv")


if not GRAPH_FILE.exists():
    print(f"{GRAPH_FILE} not found, creating graphml digraph from uberon.obo file ...")
    uberon_subset = "basic"
    uberon_url = f"http://purl.obolibrary.org/obo/uberon/{uberon_subset}.obo"
    print(f"Downloading {uberon_subset}.obo\n\tfrom: {uberon_url} ...")

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
    print(f"Graph saved\n\tto: \"{GRAPH_FILE}\".\nYou can load the graph later using: nx.read_graphml(\"{GRAPH_FILE}\")")
else:
    print(f"UBERON digraph already exists\n\tat: \"{GRAPH_FILE}\".\nLoading digraph ...")
    uberon_digraph = nx.read_graphml(GRAPH_FILE)
print(f"\t# nodes: {uberon_digraph.number_of_nodes()}\n"\
        f"\t# edges: {uberon_digraph.number_of_edges()}")

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

def get_comb(x):
    data = x[['value_x', 'value_y', 'value_z', 'master']].values.tolist()
    return list(product(*data))


if __name__ == "__main__":
    data = defaultdict(list)
    print(f"Reading .tsv files\n\tfrom: {ANNOT_DIR} ...")
    print("="*80)
    # Read annotations from all three annotators
    annot_files = [i for i in ANNOT_DIR.iterdir() if i.name.startswith("NR") and i.suffix == ".tsv"]
    for idx, file in enumerate(annot_files):
        print(f"Reading file: {file.name} ...")
        # Each file is annotation from one annotator
        pd_data = pd.read_csv(Path.joinpath(ANNOT_DIR, file), sep="\t")
        print("\tFiltering items with Entity ID following the pattern: UBERON:XXXXXXX ...")
        pd_data['Entity_ID'] = pd_data['Entity ID'].str.findall('UBERON:[0-9]{7}')
        # Keep only the rows where 'UBERON:XXXXXXX' is found
        pd_data.drop(pd_data[pd_data['Entity_ID'].str.len() == 0].index, inplace=True)
        pd_data.dropna(subset=['Entity_ID'], inplace=True)
        # Create key from character and state symbol
        print(f"\tCreating keys from Character and State Symbol ...")
        pd_data['key'] = pd_data.apply(lambda x: "{0}_{1}".format(x['Character'], x['State Symbol']), axis=1)
        # Drop all columns except newly created 'key' and 'Entity_ID'
        print("\tDropping all columns except 'key' and 'Entity_ID'")
        pd_data.drop(columns=pd_data.columns[~pd_data.columns.isin(['key', 'Entity_ID'])], inplace=True)
        # Group by 'key' and list the set of 'Entity_ID's for each key | Sort by 'key'
        temp = pd.DataFrame(
            [(group_name, list(set([j for i in group['Entity_ID'] for j in i]))) 
            for group_name, group in pd_data.groupby('key')], columns=['key', 'value'])
        temp = temp.sort_values(by="key", key=lambda x: x.str.split("_").apply(
            lambda y: (int(y[0]), int(y[1])))).reset_index(drop=True)
        data[file.stem.split("--")[-1]] = temp
        print(f"\tNumber of rows for {file.name}: {len(temp)}")
        if idx != len(annot_files) - 1:
            print("-"*80)
    print("="*80)
    # Do the same steps for Gold Standard Dataset | keep UBERON:XXXXXXX rows only | create key from character and 
    # state symbol | drop all columns expect newly created 'key' and 'Entity_ID' | group by 'key' and list the set
    # of 'Entity_ID's for each key | sort by 'key'
    gs_dataset_path = Path.joinpath(ANNOT_DIR, "GS_Dataset.tsv")
    print(f"Reading Gold Standard Dataset from :{gs_dataset_path}")
    pd_master = pd.read_csv(gs_dataset_path, sep="\t")
    print(f"\tCreating keys from Character and State Symbol ...")
    pd_master['key'] = pd_master.apply(
        lambda x: "{0}_{1}".format(x['Character'], x['State Symbol']), axis=1)
    print("\tFiltering items with Entity ID following the pattern: UBERON:XXXXXXX ...")
    pd_master['Entity_ID'] = pd_master['Entity ID'].str.findall('UBERON:[0-9]{7}')
    print("\tDropping all columns except 'key' and 'Entity_ID'")
    pd_master.drop(
        list(set(pd_master.columns) - set(['key', 'Entity_ID'])), inplace=True, axis=1)
    pd_master.dropna(subset=["Entity_ID"], inplace=True)
    pd_master = pd.DataFrame(
        [(group_name, list(set([j for i in group['Entity_ID'] for j in i]))) 
        for group_name, group in pd_master.groupby('key')], columns=['key', 'master'])
    pd_master = pd_master.sort_values(
        by="key", 
        key=lambda x: x.str.split("_").apply(
            lambda y: (int(y[0]), int(y[1])))
        ).reset_index(drop=True)
    print(f"\tNumber of rows for {gs_dataset_path.name}: {len(pd_master)}")
    print("="*80)
    print(f"Performing inner join on 'key' on all four datasets ...")
    # Merge dataframes from all three annotators on common 'key'
    vals = list(data.values())
    pd_merged = pd.merge(
        pd.merge(vals[0], vals[1], on='key', how='inner'), # Merge two dataframes on common key
        vals[2], on='key', how='inner') # merge third dataframe with the first two, again on common key
    pd_merged.columns = ['key', 'value_x', 'value_y', 'value_z']
    pd_merged = pd_merged.merge(pd_master, on='key', how='inner')
    print("Generating all possible combinations of pair of annotations, one from an annotator and another from the gold standard ...")
    pd_merged['combination'] = pd_merged.apply(get_comb, axis=1)
    print(f"Total number of rows after merging all dataset on common 'key': {len(pd_master)}")
    print(f"Saving merged dataset\n\tto: {MERGED_DATASET.parent} directory \n\tas: {MERGED_DATASET.name} ...")
    pd_merged.to_csv(MERGED_DATASET, index=False)
    print(f"Dataset successfully created.")