import os
import pickle

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from utils import pseudo_likelihood, gibbs_sampling_ising, energy, vectorize
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from minorminer import find_embedding
from copy import deepcopy
from tqdm import tqdm
from collections import namedtuple

rng = np.random.default_rng()
Instance = namedtuple("Instance", ["h", "J", "name"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()
output_path = os.path.join(ROOT, "data", "raw_data", "scalability_2d_const")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# try:
#     from config import TOKEN, TOKEN2
# except ImportError:
#     print(f"To run {__file__}, you must have \"config.py\" file with your dwave's ocean token")
#     with open(os.path.join(CWD, "config.py"), "w") as f:
#         f.write("TOKEN = \"your_ocean_token\"")


TOKEN = "julr-a86ece088ec3ae431ae7ee0541c03112c43d7af4"
if __name__ == '__main__':

    # Setup
    # qpu_sampler = DWaveSampler(solver='Advantage_system6.3', token=TOKEN2)
    qpu_sampler = DWaveSampler(
        solver="Advantage_system5.4",
        token=TOKEN,
        region="eu-central-1",
        qpu = True,
    )  # specify device used
    print(qpu_sampler.properties["extended_j_range"])
    # Experiment setup

    qpu_graph = qpu_sampler.to_networkx_graph()
    BETA_1 = 1
    NUM_SAMPLES = 100
    GIBBS_NUM_STEPS = 10**4
    ANNEAL_TIMES = [
        100,
        500,
        1000,
        2000,
    ]
    NUM_READS = 50
    ANNEAL_PARAM = 0.35  # in units of B0
    # SCALING = np.arange(2, 18, 2) # Pegasus graph size
    # print(SCALING)
    SCALING = [16] 
    PAUSE = 0
    

    # h_const = {node: 0 for node in graph.nodes}
    # J_const = {edge: -1 for edge in graph.edges}
    # const = Instance(h=h_const, J=J_const, name="constant")
    # with open(os.path.join(ROOT, "data", "p16_4.1_const.pkl"), "wb") as f:
    #     pickle.dump(const, f)
    #
    # h_uniform = {node: 0 for node in graph.nodes}
    # J_uniform = {edge: rng.uniform(-1, 1) for edge in graph.edges}
    # uniform = Instance(h=h_uniform, J=J_uniform, name="Uniform")
    # with open(os.path.join(ROOT, "data", "p16_4.1_uniform.pkl"), "wb") as f:
    #     pickle.dump(uniform, f)
    #
    # h_cbfm = {node: rng.choice([-1, 0], p=[0.85, 0.15]) for node in graph.nodes}
    # J_cbfm = {edge: rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55]) for edge in graph.edges}
    # cbfm = Instance(h=h_cbfm, J=J_cbfm, name="CBFM")
    # with open(os.path.join(ROOT, "data", "p16_4.1_cbfm.pkl"), "wb") as f:
    #     pickle.dump(cbfm, f)

    # with open(os.path.join(ROOT, "data", "p16_4.1_const.pkl"), "rb") as f:
    #     const = pickle.load(f)

    # with open(os.path.join(ROOT, "data", "p16_4.1_uniform.pkl"), "rb") as f:
    #     uniform = pickle.load(f)

    # with open(os.path.join(ROOT, "data", "p16_4.1_cbfm.pkl"), "rb") as f:
    #     cbfm = pickle.load(f)

    for m in SCALING:
        filepath = os.path.join(ROOT, "data", f"p{m}_const.pkl")
        if os.path.exists(filepath):
            print(f"loading instance P{m}")
            with open(filepath, "rb") as f:
                inst = pickle.load(f)
        else:
            print(f"generating new instance of type P{m}")
            G = dnx.pegasus_graph(m)  # source graph
            subgraph_iso = dnx.pegasus_sublattice_mappings(G, qpu_graph) # mapping to nodes of target graph
            fs = list(subgraph_iso) 
            f = fs[0] 
            Gnodes = [f(n) for n in G.nodes]
            H = qpu_graph.subgraph(Gnodes)
    
            # h_const = {node: 0 for node in H.nodes}
            # J_const = {edge: -1 for edge in H.edges}
            # inst = Instance(h=h_const, J=J_const, name="constant")
            # h_uniform = {node: 0 for node in H.nodes}
            # J_uniform = {edge: rng.uniform(-1, 1) for edge in H.edges}
            # inst = Instance(h=h_uniform, J=J_uniform, name="Uniform")
            h_cbfm = {node: rng.choice([-1, 0], p=[0.85, 0.15]) for node in H.nodes}
            J_cbfm = {edge: rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55]) for edge in H.edges}
            inst = Instance(h=h_cbfm, J=J_cbfm, name="const")
            with open(os.path.join(ROOT, "data", filepath), "wb") as f:
                pickle.dump(inst, f)

        h = inst.h
        J = inst.J
        chain = nx.Graph(J.keys())
        chain_length = len(h)

        for anneal_time in ANNEAL_TIMES:

            E_fin = []
            configurations = []
            Q = []
            raw_data = pd.DataFrame(
                columns=["qubits", "sample", "energy", "num_occurrences", "init_state"]
            )
            for i in tqdm(
                range(NUM_SAMPLES),
                desc=f"samples for P{m} with {chain_length} qubits, anneal time {anneal_time:.2f} s",
            ):
                initial_state = dict(
                    gibbs_sampling_ising(h, J, BETA_1, GIBBS_NUM_STEPS)
                )

                # anneal_schedule = [[0, 1], [ANNEAL_TIME * 1 / 2 - pause_duration / 2, ANNEAL_PARAM],
                #                    [ANNEAL_TIME * 1 / 2 + pause_duration / 2, ANNEAL_PARAM],
                #                    [ANNEAL_TIME, 1]] if pause_duration != 0 else \
                #     [[0, 1], [ANNEAL_TIME / 2, ANNEAL_PARAM], [ANNEAL_TIME, 1]]
                anneal_schedule = [
                    [0, 1],
                    [anneal_time / 2, ANNEAL_PARAM],
                    [anneal_time, 1],
                ]

                # sampler = EmbeddingComposite(qpu_sampler)
                sampler = qpu_sampler

                sampleset = sampler.sample_ising(
                    h=h,
                    J=J,
                    initial_state=initial_state,
                    anneal_schedule=anneal_schedule,
                    num_reads=NUM_READS,
                    auto_scale=True,
                    reinitialize_state=True,
                )

                df = sampleset.to_pandas_dataframe(sample_column=True)
                df["init_state"] = [initial_state for _ in range(len(df))]
                raw_data = pd.concat([raw_data, df], ignore_index=True)

                raw_data.to_csv(
                    os.path.join(
                        output_path,
                        f"raw_data_pegasus_{m}_{anneal_time}.csv",
                    ),
                )
