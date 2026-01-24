import os
import pickle
import gc  # for garbage collection to prevent memory leaks

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from utils import energy, vectorize_2d, gibbs_sampling_efficient
from dwave.system.samplers import DWaveSampler
from minorminer import find_embedding
from tqdm import tqdm
from collections import namedtuple
import dimod
import glob  # for file pattern matching

import warnings
# suppress FutureWarning from pandas
warnings.simplefilter("ignore", FutureWarning)

rng = np.random.default_rng()
Instance = namedtuple("Instance", ["h", "J", "name"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()


if __name__ == "__main__":

    # This part may need changes depending on how you comunicate with your machine
    qpu_sampler = DWaveSampler(
        solver="Advantage_system6.4",
        # solver="Advantage2_system1.6",
    )  # specify device used
    target = qpu_sampler.to_networkx_graph()

    BETA_1_VALUES = [round(s, 3) for s in  list(np.arange(0.5, 1.1, 0.1))]
    print(f"BETA_1_VALUES: {BETA_1_VALUES}")
    NUM_SAMPLES = 50
    GIBBS_NUM_SWEEPS = 10**3
    anneal_time = [100] #, 200, 500, 1000]
    NUM_READS = 50
    initial_value = 1.0
    anneal_params = {
        "CON": [round(s, 3) for s in np.arange(0.1, 0.9, 0.05)],
        # "RAU": np.arange(0.1, 0.9, 0.01),
        # "CBFM": np.arange(0.1, 0.9, 0.01),
    }
    
    SCALING = [6]  # crashed for 1000 at 382 samples

    itype = ["CON"]
    instance_graph = "pegasus"

    print("Starting processing of instances...")
    print("beta_1_values:", BETA_1_VALUES)
    print("anneal_time:", anneal_time)
    for BETA_1 in BETA_1_VALUES:
        for chain_length in SCALING:
            for ITYPE in itype:
                # Find all instances of the given type and chain length in pegasus_native
                pattern = os.path.join(
                    ROOT, "data", "instances", f"sub{instance_graph}_native", f"P{chain_length}_{ITYPE}_*.pkl"
                )
                instance_files = sorted(glob.glob(pattern))
                
                if not instance_files:
                    print(f"No instances found for P{chain_length} type {ITYPE} in sub{instance_graph}_native directory")
                    continue
                
                print(f"Found {len(instance_files)} instances for P{chain_length} type {ITYPE}")
                
                # Process each instance file
                for inst_idx, instance_file in enumerate(instance_files):
                    instance_basename = os.path.basename(instance_file).replace('.pkl', '')
                    
                    output_path = os.path.join(
                        # ROOT, "data", "raw_data", f"fixed_gibbs_annealing_param_P{chain_length}_{ITYPE}_advantage6.4"
                        ROOT, "data", "raw_data", f"fixed_gibbs_phase_diagram_P{chain_length}_{ITYPE}_advantage6.4"
                    )
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
                    print(f"Processing instance {inst_idx + 1}/{len(instance_files)}: {instance_basename} with BETA_1={BETA_1}")
                    
                    # Load instance from pegasus_native
                    with open(instance_file, "rb") as f:
                        inst_data = pickle.load(f)
                    
                    # Convert from list format [h, J] to Instance namedtuple
                    if isinstance(inst_data, list) and len(inst_data) == 2:
                        h = inst_data[0]
                        J = inst_data[1]
                        inst = Instance(h=h, J=J, name=f"{instance_basename}_beta_{BETA_1:.1f}")
                    else:
                        print(f"Unexpected instance format in {instance_file}")
                        continue
                    h = inst.h
                    J = inst.J
                    h_vect, J_vect, h_new, J_new, _ = vectorize_2d(h, J)

                    # for pause_duration in PAUSES:
                    for ANNEAL_TIME in anneal_time:
                        for ANNEAL_PARAM in anneal_params[ITYPE]:
                            E_fin = []
                            configurations = []
                            Q = []
                            raw_data = pd.DataFrame(
                                columns=["sample", "energy", "num_occurrences", "init_state"]
                            )
                            results_path = os.path.join(
                                output_path,
                                f"raw_data_{instance_basename}_beta_{BETA_1:.1f}_{ANNEAL_TIME}_{ANNEAL_PARAM:.3f}.csv",
                            )
                            if os.path.exists(results_path):
                                with open(results_path, "r") as file:
                                    line_count = sum(1 for _ in file)
                                if line_count > NUM_SAMPLES * NUM_READS * 0.9:
                                    print(
                                        f"results already exist for {instance_basename}, beta={BETA_1}, at={ANNEAL_TIME}, ap={ANNEAL_PARAM} with sufficient data"
                                    )
                                    continue
                            for i in tqdm(
                                range(NUM_SAMPLES),
                                desc=f"samples for {instance_basename} beta {BETA_1:.1f} anneal time {ANNEAL_TIME:.2f} micro s and anneal param {ANNEAL_PARAM:.2f}",
                            ):
                               
                                # tmp = gibbs_sampling_ising_vectorized_2d(h, J, BETA_1, GIBBS_NUM_STEPS)
                                # initial_state = {k: v for k, v in zip(h_new.keys(), tmp)}
                                # for i, (k,v) in enumerate(initial_state.items()):
                                #   print(f"{i}: {k} -> {v}, {tmp[i]}")
                                # exit(1)
                                initial_state = dict(
                                 gibbs_sampling_efficient(h, J, BETA_1, GIBBS_NUM_SWEEPS)
                                )
                               
                                init_state = np.array(list(initial_state.values()))
                                E_init = energy(init_state, h_vect, J_vect) / len(h)

                                anneal_schedule = [
                                    [0.0, initial_value],
                                    [ANNEAL_TIME / 2, ANNEAL_PARAM],
                                    [ANNEAL_TIME, initial_value],
                                ]
                                try:
                                  sampler = (qpu_sampler)
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

                                  raw_data.to_csv(results_path)
                                  
                                  # Memory management: explicitly delete large objects
                                  del df, sampleset
                                  gc.collect()
                                  
                                except Exception as e:
                                  print(f"Error: {e}")
                                  continue
                    
                    # Clean up memory after processing each instance
                    del h_vect, J_vect, h_new, J_new, h, J, inst
                    gc.collect()
                    print(f"Completed processing {instance_basename} with BETA_1={BETA_1}")
                    
    print("All processing completed!")
