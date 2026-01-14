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

# Import grid sampling utilities for parameter generation
from grid_sampling import Line, generate_parameter_pairs_with_separate_densities 

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
        solver="Advantage_system6.4",
        token = "julr-bf16fdadab879dbeb1960fe55070031134855957",
    )  # specify device used
    target = qpu_sampler.to_networkx_graph()

    # Define parameter pairs (BETA_1, ANNEAL_PARAM) using grid-based sampling
    
    # Option 1: Rectangular parameter space using grid sampling
    # PARAMETER_PAIRS, grid_info = generate_parameter_pairs_grid(
    #     beta_range=(0.5, 3.0),
    #     anneal_range=(0.1, 0.9),
    #     density=8  # Creates an 8x8 grid in the rectangular region
    # )

    line1 = Line.from_two_points((2.0, 0.1), (0.5, 0.6))
    line2 = Line.from_two_points((3.0, 0.1), (1.0, 0.9))
    
    point_data = generate_parameter_pairs_with_separate_densities(
      beta_range=(0.5, 3.0),
      anneal_range=(0.1, 0.9),
      inside_density=(30, 30),
      outside_density=(15, 15),
      boundary_lines=(line1, line2),
      sample_both=True  
    )
    print(point_data.keys())
    PARAMETER_PAIRS = point_data["all_pairs"]

    
    print(f"Generated {len(PARAMETER_PAIRS)} parameter pairs using grid-based sampling:")
    print(f"  - Total pairs: {len(PARAMETER_PAIRS)}")
    
    # Display sample of generated pairs
    print(f"\nSample parameter pairs:")
    for i, (beta, anneal) in enumerate(PARAMETER_PAIRS[:10]):
        print(f"  ({beta:.2f}, {anneal:.2f})")
    if len(PARAMETER_PAIRS) > 10:
        print(f"  ... and {len(PARAMETER_PAIRS) - 10} more pairs")
    
    # Optional: Create visualization of parameter space coverage
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot grid-based pairs
        grid_pairs = PARAMETER_PAIRS
        if grid_pairs:
            betas_grid = [p[0] for p in grid_pairs]
            anneals_grid = [p[1] for p in grid_pairs]
            ax.scatter(anneals_grid, betas_grid, c='blue', alpha=0.7, s=50, 
                      label=f'Grid pairs ({len(grid_pairs)})', marker='o')
        
        ax.set_ylabel('BETA_1')
        ax.set_xlabel('ANNEAL_PARAM')
        ax.set_title('Parameter Space Coverage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(ROOT, 'plots', 'parameter_space_coverage.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nParameter space visualization saved to: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"Could not create parameter space visualization: {e}")
    
    print(f"\nProcessing {len(PARAMETER_PAIRS)} parameter pairs")
    
    NUM_SAMPLES = 50
    GIBBS_NUM_STEPS = 10**4
    anneal_time = [100]
    NUM_READS = 50
    initial_value = 1.0
    SCALING = [6]  # crashed for 1000 at 382 samples

    # itype = ["const", "unif", "cbfm"]
    itype = ["const"]

    for BETA_1, ANNEAL_PARAM in PARAMETER_PAIRS:
        print(f"Processing parameter pair: BETA_1={BETA_1}, ANNEAL_PARAM={ANNEAL_PARAM}")
        
        for chain_length in SCALING:
            for ITYPE in itype:
                # Find all instances of the given type and chain length in pegasus_native
                pattern = os.path.join(
                    ROOT, "data", "instances", "pegasus_native", f"P{chain_length}_{ITYPE}*.pkl"
                )
                instance_files = sorted(glob.glob(pattern))
                instance_files = [f for f in instance_files if 'large_h'  not in f]  # filter for large_h instances
                
                if not instance_files:
                    print(f"No instances found for P{chain_length} type {ITYPE} in pegasus_native directory")
                    continue
                
                print(f"Found {len(instance_files)} instances for P{chain_length} type {ITYPE}")
                
                # Process each instance file
                for inst_idx, instance_file in enumerate(instance_files):
                    instance_basename = os.path.basename(instance_file).replace('.pkl', '')
                    
                    output_path = os.path.join(
                        ROOT, "data", "raw_data", f"phase_diagram_P{chain_length}_{ITYPE}_advantage_6.4_final"
                    )
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
                    print(f"Processing instance {inst_idx + 1}/{len(instance_files)}: {instance_basename} with BETA_1={BETA_1}, ANNEAL_PARAM={ANNEAL_PARAM}")
                    
                    # Load instance from pegasus_native
                    with open(instance_file, "rb") as f:
                        inst_data = pickle.load(f)
                    
                    # Convert from list format [h, J] to Instance namedtuple
                    if isinstance(inst_data, list) and len(inst_data) == 2:
                        h = inst_data[0]
                        J = inst_data[1]
                        inst = Instance(h=h, J=J, name=f"{instance_basename}_beta_{BETA_1:.5f}_ap_{ANNEAL_PARAM:.5f}")
                    else:
                        print(f"Unexpected instance format in {instance_file}")
                        continue
                    h = inst.h
                    J = inst.J
                    h_vect, J_vect, h_new, J_new, _ = vectorize_2d(h, J)

                    # for pause_duration in PAUSES:
                    for ANNEAL_TIME in anneal_time:
                        E_fin = []
                        configurations = []
                        Q = []
                        raw_data = pd.DataFrame(
                            columns=["sample", "energy", "num_occurrences", "init_state"]
                        )
                        results_path = os.path.join(
                            output_path,
                            f"raw_data_{instance_basename}_beta_{BETA_1:.5f}_{ANNEAL_TIME}_{ANNEAL_PARAM:.5f}.csv",
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
                            desc=f"samples for {instance_basename} beta {BETA_1:.5f} anneal time {ANNEAL_TIME:.5f} micro s and anneal param {ANNEAL_PARAM:.5f}",
                        ):
                           
                            # tmp = gibbs_sampling_ising_vectorized_2d(h, J, BETA_1, GIBBS_NUM_STEPS)
                            # initial_state = {k: v for k, v in zip(h_new.keys(), tmp)}
                            # for i, (k,v) in enumerate(initial_state.items()):
                            #   print(f"{i}: {k} -> {v}, {tmp[i]}")
                            # exit(1)
                            if BETA_1 <= 10:
                              GIBBS_NUM_STEPS = 10**4
                            else:
                              GIBBS_NUM_STEPS = 10**5
                            initial_state = dict(
                             gibbs_sampling_efficient(h, J, BETA_1, GIBBS_NUM_STEPS) 
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
                print(f"Completed processing {instance_basename} with BETA_1={BETA_1}, ANNEAL_PARAM={ANNEAL_PARAM}")
                    
    print("All processing completed!")