import copy
import os
import json
import glob
import minorminer

import numpy as np
import networkx as nx
import dwave_networkx as dnx

from dwave.system import DWaveSampler
from dwave.embedding import is_valid_embedding
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Optional
from math import ceil

from src.utils import find_embedding

rng = np.random.default_rng()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DIR_BASE = os.path.join(ROOT, "data", "Advantage_system5.4")


def find_disjoint_k_cliques(clique_size: int, graph: nx.Graph):
    # using networkx clique finder.  Empirically this seems fast enough to not need to code up a
    # special purpose k-clique finder taking advantage of node degree or an analytical solution
    # for a given solver topology.
    cliques = nx.find_cliques(graph)
    cliques = sorted(map(sorted, [c for c in cliques if len(c) == clique_size]))

    used_nodes = set()
    disjoint_cliques = []
    for clique in cliques:
        if not any(map(lambda n: n in used_nodes, clique)):
            disjoint_cliques.append(clique)
            used_nodes.update(clique)
    return disjoint_cliques


def find_minimal_num_unit_cells(graph: nx.Graph):
    n = graph.number_of_nodes()
    return int(ceil(n/24))


def tile_instance(graph: nx.Graph, qpu_graph: nx.Graph):
    embedded_instances = []
    unit_cells = []
    pegasus_nice_numbering = {node: dnx.pegasus_coordinates(16).linear_to_nice(node) for node in graph.nodes}
    lattice_size = 15
    xs = range(lattice_size)
    ys = range(lattice_size)
    min_unit_cells = find_minimal_num_unit_cells(graph)
    for x_bound in tqdm(xs, desc="finding partition of QPU graph"):
        ...

def remap_king_ladders_pegasus(graph: nx.Graph, instances: list[dict]):
    embedded_instances = []
    unit_cells = []
    pegasus_nice_numbering = {node: dnx.pegasus_coordinates(16).linear_to_nice(node) for node in graph.nodes}
    lattice_size = 15
    xs = range(lattice_size)
    ys = range(lattice_size)
    for x_bound in tqdm(xs, desc="finding unit cells"):
        for y_bound in ys:
            unit_cell = copy.deepcopy(graph)
            for node in graph.nodes():
                t, y, x, u, k = pegasus_nice_numbering[node]
                if y != y_bound or x != x_bound:
                    unit_cell.remove_node(node)
            assert 0 < len(unit_cell) <= 24
            unit_cells.append(unit_cell)
    used_unit_cells = rng.choice(list(range(len(unit_cells))), size=len(instances), replace=False)
    for unit_cell_index, instance in tqdm(zip(used_unit_cells, instances), desc="finding embeddings"):
        J = {(quad["id_tail"], quad["id_head"]): quad["coeff"] for quad in instance["quadratic_terms"]}
        source_graph = nx.Graph()
        source_graph.add_edges_from(list(J.keys()))
        embedding = find_embedding(source_graph, unit_cells[unit_cell_index])
        assert is_valid_embedding(embedding, source_graph, graph)
        embedding = {k: v[0] for k,v in embedding.items()}
        embedded_instances.append(remap_instance(instance, [embedding[v] for v in source_graph.nodes], embedding))
    return embedded_instances


def get_spin_system_bqpjson(instance):
    system = []
    for linear_term in instance['linear_terms']:
        system.append({"i": linear_term['id'], "v": linear_term['coeff']})
    for quadratic_term in instance['quadratic_terms']:
        system.append({"i": quadratic_term['id_tail'], "j": quadratic_term['id_head'], "v": quadratic_term['coeff']})
    spin_system = {'spins': list(sorted(instance['variable_ids'])), 'values': system}
    return spin_system


def write_instances_to_file(instances, outfile):
    spin_systems = [get_spin_system_bqpjson(instance) for instance in instances]
    linear_terms = []
    quadratic_terms = []
    variable_ids = []

    for instance in instances:
        linear_terms = linear_terms + instance['linear_terms']
        quadratic_terms = quadratic_terms + instance['quadratic_terms']
        variable_ids = variable_ids + instance['variable_ids']

    variable_ids.sort()
    bqp_instance = {
        "id": len(instances),
        "version": "1.0.0",
        "description": "Tiled Hamiltonians to be run on D-Wave",
        "variable_domain": "spin",
        "scale": 1.0,
        "offset": 0.0,
        "linear_terms": linear_terms,
        "quadratic_terms": quadratic_terms,
        "variable_ids": variable_ids,
        "metadata": {
            "spin_systems": spin_systems
        }
    }

    json_instance = json.dumps(bqp_instance, indent=2)
    with open(outfile, "w") as f:
        f.write(json_instance)


def read_bqpjson_instance(filepath):
    with open(filepath) as file:
        bqpjson = json.load(file)
    return bqpjson


def remap_instance(instance, new_labels, embedding: Optional = None):
    ids = instance["variable_ids"]
    assert len(ids) <= len(new_labels), "System size must be smaller than clique found on lattice. \
                                        Given: " + str(len(ids)) + "; Maximum: " + str(len(new_labels))
    linear_terms = instance["linear_terms"]
    quadratic_terms = instance["quadratic_terms"]
    if embedding is None:
        id_map = {ids[i]: new_labels[i] for i in range(len(ids))}
    else:
        id_map = embedding
    new_instance = {}
    remapped_linear_terms = []
    remapped_quadratic_terms = []
    for term in linear_terms:
        remapped_linear_terms.append({"id": id_map[term["id"]],
                                      "coeff": term["coeff"]})

    for term in quadratic_terms:
        remapped_quadratic_terms.append({"id_tail": id_map[term["id_tail"]],
                                         "id_head": id_map[term["id_head"]],
                                         "coeff": term["coeff"]})

    new_instance["linear_terms"] = remapped_linear_terms
    new_instance["quadratic_terms"] = remapped_quadratic_terms
    new_instance["variable_ids"] = new_labels
    return new_instance


if __name__ == '__main__':
    instance_type = "king_ladder"
    size = 8
    # instance_type = "chain"
    # size = 4

    instances_dir = os.path.join(INSTANCE_DIR_BASE, f"{instance_type}s", f"{instance_type}_{size}")
    out_file = os.path.join(INSTANCE_DIR_BASE, f"{instance_type}s", f"tiled_{instance_type}_{size}.json")

    qpu_sampler = DWaveSampler(solver="Advantage_system6.4")
    qpu_graph = qpu_sampler.to_networkx_graph()

    # cliques = find_disjoint_k_cliques(4, qpu_graph)
    # cliques3 = [[a,b,c] for [a,b,c,_] in cliques]
    instance_files = glob.glob(instances_dir + "/*.json")
    instances = [read_bqpjson_instance(instance_file) for instance_file in instance_files]

    # random_cliques = rng.choice(list(range(len(cliques))), size=len(instances), replace=False)

    # embedded_instances = [remap_instance(instances[i], cliques[random_cliques[i]])
    #                       for i in range(len(instances))]


    embedded_instances = remap_king_ladders_pegasus(qpu_graph, instances)
    write_instances_to_file(embedded_instances, out_file)