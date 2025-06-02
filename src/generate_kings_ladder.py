import os
import pickle

import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np

from dwave.system import DWaveSampler
from itertools import product
from minorminer import find_embedding
from tqdm import tqdm
from typing import Optional

rng = np.random.default_rng()


def create_kings_ladder(size: int) -> nx.Graph:
    """
    Create kings ladder with nodes labelled from 0 to 2*size-1
    :param size: Number of "rugs" on the ladder
    :return: nx.Graph
    """
    graph = nx.ladder_graph(size)
    for i in range(size-1):
        graph.add_edge(i, i + size + 1)
    for i in range(size, 2 * size - 1):
        graph.add_edge(i, i - size + 1)
    return graph


def find_disjoint_4_cliques(topology_graph: nx.Graph) -> list:
    cliques = nx.find_cliques(topology_graph)
    cliques = sorted(map(sorted, [c for c in cliques if len(c) == 4]))

    used_nodes = set()
    disjoint_cliques = []
    for clique in cliques:
        if not any(map(lambda n: n in used_nodes, clique)):
            disjoint_cliques.append(clique)
            used_nodes.update(clique)
    return disjoint_cliques


def create_clique_graph(topology_graph: nx.Graph) -> nx.Graph:
    cliques = find_disjoint_4_cliques(topology_graph)
    clique_graph = nx.Graph()

    for clique in cliques:
        clique_graph.add_nodes_from(clique)

    for edge in product(list(clique_graph.nodes), repeat=2):
        if edge in topology_graph.edges:
            clique_graph.add_edge(*edge)

    return clique_graph


def create_random_instance(graph: nx.Graph, no_external_fields: bool = False):
    h = {node: 0 if no_external_fields else rng.uniform(-1, 1) for node in graph.nodes}
    J = {edge: rng.uniform() for edge in graph.edges}
    return J, h


def save_instance_coo(couplings: dict, bias: dict, save_path: os.PathLike):
    with open(save_path, "w") as f:
        for node, value in bias.items():
            f.write(f"{node} {node} {value.item()}\n")
        for (e1, e2), value in couplings.items():
            f.write(f"{e1} {e2} {value.item()}\n")


def save_instance_pickle(couplings: dict, bias: dict, save_path: str):
    data = [couplings, bias]
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def generate_embedded_kings_ladders(
        number: int,
        size: int,
        sampler_graph: nx.Graph,
        path: str,
        category: str = "random",
        name: Optional[str] = None) -> None:

    print("Preprocessing")
    mapping = {node: dnx.pegasus_coordinates(16).nice_to_linear(node) for node in sampler_graph}
    nice_clique_graph = create_clique_graph(sampler_graph)
    clique_graph = nx.relabel_nodes(nice_clique_graph, mapping)
    ladder = create_kings_ladder(size)

    for i in tqdm(range(number), desc=f"Generating kings ladder instances of size {size}"):
        embedding = find_embedding(ladder, clique_graph)

        assert max([len(chain) for chain in embedding.values()]) == 1
        if category == "random":
            J, h = create_random_instance(ladder)
        else:
            raise ValueError("Wrong value of \"category\"")

        J_embedded = {(embedding[v][0], embedding[w][0]): value for (v,w), value in J.items()}
        h_embedded = {embedding[k][0]: value for k, value in h.items()}

        name = "kings_ladder_" if name is None else name
        save_path = os.path.join(path, name + f"{i+1}.pkl")
        save_instance_pickle(J_embedded, h_embedded, save_path)



