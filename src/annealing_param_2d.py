import os
import pickle
import dwave.inspector

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from utils import pseudo_likelihood, gibbs_sampling_ising_vectorized_2d, gibbs_sampling_ising, energy, vectorize_2d
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
from minorminer import find_embedding
from copy import deepcopy
from tqdm import tqdm
from collections import namedtuple
import dimod

import warnings
# suppress FutureWarning from pandas
warnings.simplefilter("ignore", FutureWarning)

rng = np.random.default_rng()
Instance = namedtuple("Instance", ["h", "J", "name"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()


# try:
#     from config import TOKEN
# except ImportError:
#     print(f"To run {__file__}, you must have \"config.py\" file with your dwave's ocean token")
#     with open(os.path.join(CWD, "config.py"), "w") as f:
#         f.write("TOKEN = \"your_ocean_token\"")

# small h [-0.5, 0.5], medium h [-1, 1], large h [-2, 2]


if __name__ == "__main__":

    # This part may need changes depending on how you comunicate with your machine
    qpu_sampler = DWaveSampler(
        solver="Advantage_system5.4",
        token="julr-a86ece088ec3ae431ae7ee0541c03112c43d7af4",
        region="eu-central-1",
    )  # specify device used
    target = qpu_sampler.to_networkx_graph()

    BETA_1 = 1
    NUM_SAMPLES = 50
    GIBBS_NUM_STEPS = 10**4
    # anneal_time = [5, 50]  # crashed for 500 at 89 samples
    # anneal_time = [100, 200]  # crashed for 500 at 89 samples
    # anneal_time = [500, 1000]  # crashed for 500 at 89 samples
    anneal_time = [5, 50, 100, 200, 500, 1000]
    NUM_READS = 200
    initial_value = 1.0
    # ANNEAL_PARAM = 2.96227 / 8.586335
    anneal_param = np.arange(0.1, 1.0, 0.1)
    SCALING = [4]  # crashed for 1000 at 382 samples

    # itype = ["const", "unif", "cbfm"]
    itype = ["const"]

    for chain_length in SCALING:
        for ITYPE in itype:
            output_path = os.path.join(
                ROOT, "data", "raw_data", f"old_gibbs_annealing_param_2d_{ITYPE}"
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filepath = os.path.join(
                ROOT, "data", "instances", f"p{chain_length}_{ITYPE}.pkl"
            )
            if os.path.exists(filepath):
                print(f"loading instance of type P{chain_length} and type {ITYPE}")
                with open(filepath, "rb") as f:
                    inst = pickle.load(f)
            else:
                print(
                    f"generating new instance of type P{chain_length} and type {ITYPE}"
                )
                G = dnx.pegasus_coordinates(chain_length).graph_to_linear(
                    dnx.pegasus_graph(chain_length, nice_coordinates=True)
                )
                # subgraph_iso = dnx.pegasus_sublattice_mappings(G, target) # mapping to nodes of target graph
                # fs = list(subgraph_iso)
                # f = fs[0]
                # Gnodes = [f(n) for n in G.nodes]
                # H = target.subgraph(Gnodes)
                if ITYPE == "const":
                    h_const = {node: 0 for node in G.nodes}
                    J_const = {edge: -1 for edge in G.edges}
                    inst = Instance(h=h_const, J=J_const, name="constant")
                if ITYPE == "unif":
                    h_uniform = {node: 0 for node in G.nodes}
                    J_uniform = {edge: rng.uniform(-1, 1) for edge in G.edges}
                    inst = Instance(h=h_uniform, J=J_uniform, name="uniform")
                if ITYPE == "cbfm":
                    h_cbfm = {
                        node: rng.choice([-1, 0], p=[0.85, 0.15]) for node in G.nodes
                    }
                    J_cbfm = {
                        edge: rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55])
                        for edge in G.edges
                    }
                    inst = Instance(h=h_cbfm, J=J_cbfm, name="cbfm")

                with open(os.path.join(ROOT, "data", filepath), "wb") as f:
                    pickle.dump(inst, f)
            h = inst.h
            J = inst.J
            h_vect, J_vect, h_new, J_new, _ = vectorize_2d(h, J)

            # for pause_duration in PAUSES:
            for ANNEAL_TIME in anneal_time:
                for ANNEAL_PARAM in anneal_param:
                    E_fin = []
                    configurations = []
                    Q = []
                    raw_data = pd.DataFrame(
                        columns=["sample", "energy", "num_occurrences", "init_state"]
                    )
                    results_path = os.path.join(
                        output_path,
                        f"raw_data_pegasus_{chain_length}_{ANNEAL_TIME}_{ANNEAL_PARAM:.3f}.csv",
                    )
                    if os.path.exists(results_path):
                        with open(results_path, "r") as file:
                            line_count = sum(1 for _ in file)
                        if line_count > 8000:
                            print(
                                f"results already exist for cl={chain_length}, at={ANNEAL_TIME}, ap={ANNEAL_PARAM}, b1={BETA_1} with sufficient data"
                            )
                            continue
                    for i in tqdm(
                        range(NUM_SAMPLES),
                        desc=f"samples for {chain_length} anneal time {ANNEAL_TIME:.2f} micro s and anneal param {ANNEAL_PARAM:.2f}",
                    ):
                       
                        # tmp = gibbs_sampling_ising_vectorized_2d(h, J, BETA_1, GIBBS_NUM_STEPS)
                        # initial_state = {k: v for k, v in zip(h_new.keys(), tmp)}
                        # for i, (k,v) in enumerate(initial_state.items()):
                        #   print(f"{i}: {k} -> {v}, {tmp[i]}")
                        # exit(1)
                        initial_state = dict(
                         gibbs_sampling_ising(h_new, J_new, BETA_1, GIBBS_NUM_STEPS) 
                        )
                       
                        init_state = np.array(list(initial_state.values()))
                        E_init = energy(init_state, h_vect, J_vect) / len(h)

                        anneal_schedule = [
                            [0.0, initial_value],
                            [ANNEAL_TIME / 2, ANNEAL_PARAM],
                            [ANNEAL_TIME, initial_value],
                        ]
                        try:
                          sampler = EmbeddingComposite(qpu_sampler)
                          sampleset = sampler.sample_ising(
                              h=h_new,
                              J=J_new,
                              initial_state=initial_state,
                              anneal_schedule=anneal_schedule,
                              num_reads=NUM_READS,
                              auto_scale=True,
                              reinitialize_state=True,
                          )

                          df = sampleset.to_pandas_dataframe(sample_column=True)
                          df["init_state"] = [initial_state for _ in range(len(df))]
                          raw_data = pd.concat([raw_data, df], ignore_index=True)

                          raw_data.to_csv(results_path)
                        except Exception as e:
                          print(f"Error: {e}")
                          continue
