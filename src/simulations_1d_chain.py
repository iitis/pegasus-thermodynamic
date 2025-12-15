import os
import pickle
import gc
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from utils import gibbs_sampling_ising, energy, vectorize
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from tqdm import tqdm

rng = np.random.default_rng()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()



if __name__ == "__main__":

    qpu = "pegasus"

    # This part may need changes depending on how you comunicate with your machine
    qpu_sampler = DWaveSampler(
        solver="Advantage2_system1.6" if qpu == "zephyr" else "Advantage_system6.4",
        token="julr-bf16fdadab879dbeb1960fe55070031134855957",
    )  # specify device used
    target = qpu_sampler.to_networkx_graph()

    NUM_SAMPLES = 200
    GIBBS_NUM_STEPS = 10**4
    anneal_time = [100, 500]  # crashed for 500 at 89 samples
    NUM_READS = 50
    initial_value = 1.0
    BETA_1s = [1.0]
    BETA_1s = [np.round(b, 3) for (i, b) in enumerate(BETA_1s)]
    print(f"beta_1s: {BETA_1s}")
    anneal_param = [0.75]

    CHAIN_LENGHTS = [2000, 3000]
    hrange = [2.0]
    hname = ["unif_large_h"]

    for chain_length in CHAIN_LENGHTS:
        for mag_field, name in zip(hrange, hname):
            output_path = os.path.join(ROOT, "data", "raw_data", f"scalability_{qpu}_1d_chain_L_{chain_length}_{name}_{'advantage_6.4' if qpu == 'pegasus' else 'advantage2_1.6'}")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filepath = os.path.join(
                ROOT, "data", "instances", f"1d_{qpu}_chains", f"{qpu}_1D_chain_L_{chain_length}_{name}.pkl"
            )
            if os.path.exists(filepath):
                print(f"loading instance of length {chain_length}, {name}")
                with open(filepath, "rb") as f:
                    h, J = pickle.load(f)
            else:
                print(f"generating new instance of length {chain_length}, {name}")
                h = {
                    node: rng.uniform(-mag_field, mag_field)
                    for node in range(chain_length)
                }
                J = {
                    (node, node + 1): rng.uniform(-1, 1)
                    for node in range(chain_length - 1)
                }
                with open(filepath, "wb") as f:
                    l = [h, J]
                    pickle.dump(l, f)

            h_vect, J_vect = vectorize(h, J)
            chain = nx.Graph(J.keys())

            # for pause_duration in PAUSES:
            for ANNEAL_TIME in anneal_time:
              for ANNEAL_PARAM in anneal_param:
                for BETA_1 in BETA_1s:
                  E_fin = []
                  configurations = []
                  Q = []
                  raw_data = pd.DataFrame(
                      columns=["sample", "energy", "num_occurrences", "init_state"]
                  )
                  results_path = os.path.join(
                      output_path,
                      f"raw_data_chain_{chain_length}_{ANNEAL_TIME}_{ANNEAL_PARAM:.3f}_{BETA_1:.3f}.csv",
                  )
                  if os.path.exists(results_path):
                      with open(results_path, "r") as file:
                          line_count = sum(1 for _ in file)
                      if line_count > 8000:
                          print(
                              f"results already exist for cl={chain_length}, at={ANNEAL_TIME}, ap={ANNEAL_PARAM}, b1={BETA_1} with sufficient data"
                          )
                          continue
                  print(f"samples for chain len {chain_length}, anneal time {ANNEAL_TIME:.2f} micro s and anneal param {ANNEAL_PARAM:.2f}, beta_1 {BETA_1:.2f}")
                  for i in tqdm(
                      range(NUM_SAMPLES),
                      desc="samples: "
                  ):
                      initial_state = dict(
                          gibbs_sampling_ising(h, J, float(BETA_1), GIBBS_NUM_STEPS)
                      )
                      # print(initial_state)
                      init_state = np.array(list(initial_state.values()))

                      E_init = (
                          energy(init_state, h_vect, J_vect) / chain_length
                      )  # per spin
              
                      anneal_schedule = (
        
                          [
                              [0.0, initial_value],
                              [ANNEAL_TIME / 2, ANNEAL_PARAM],
                              [ANNEAL_TIME, initial_value],
                          ]
                      )

                      try:
                        sampler = EmbeddingComposite(qpu_sampler)

                        sampleset = sampler.sample_ising(
                            h=h,
                            J=J,
                            initial_state=initial_state,
                            anneal_schedule=anneal_schedule,
                            num_reads=NUM_READS,
                            auto_scale=False,
                            reinitialize_state=True,
                        )

                        df = sampleset.to_pandas_dataframe(sample_column=True)
                        df["init_state"] = [initial_state for _ in range(len(df))]
                        raw_data = pd.concat([raw_data, df], ignore_index=True)

                        raw_data.to_csv(
                            os.path.join(
                                output_path,
                                results_path
                            )
                        )
                        del df, sampleset
                        gc.collect()
                      except Exception as e:
                          print(f"An error occurred: {e}")
                          continue
