import os
import pickle

import numpy as np
import pandas as pd

from scipy import optimize
from utils import (
    pseudo_likelihood_2d_vectorised,
    extend,
    vectorize,
    energy,
    vectorize_2d,
)
from collections import namedtuple
from dimod import BinaryQuadraticModel

Instance = namedtuple("Instance", ["h", "J", "name"])

PHYSICAL_UNITS = False

B0 = 1.0  # actually B0 = 8.58.. but this is included into annealing schedule
h = 6.62607015e-34  # Planck constant, in J/Hz
kb = 1.380649e-23  # Boltzmann constant, in J/K
energy_units = (B0 / 2) * 10**9 * h
beta_units = kb / energy_units
if PHYSICAL_UNITS == False:
    energy_units = 1
    beta_units = 1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()
# DATA = os.path.join(ROOT, "data", "raw_data", "scalability_2d_cbfm")
# RESULTS = os.path.join(ROOT, "data", "results", "scalability_2d_cbfm")


def main():
    # Process P6 const instances from phase diagram data
    inst_types = ["CON", "RAU", "CBFM"]
    for inst_type in inst_types:
        print(f"Running optimization for annealing param P6 {inst_type} data")
        DATA = os.path.join(
            ROOT, "data", "raw_data", f"annealing_param_P6_{inst_type}_advantage6.4"
        )
        RESULTS = os.path.join(
            ROOT, "data", "results", "paper", f"annealing_param_P6_{inst_type}_advantage6.4"
        )

        if not os.path.exists(RESULTS):
            os.makedirs(RESULTS)

        betas = {}
        energies = {}
        Q = {}
        Q_dist = {}

        # New filename format: raw_data_P6_const_<instance_num>_beta_<beta>_<anneal_time>_<anneal_param>.csv
        sort_key = lambda x: (
            # int(x.split("_")[6]),
            # float(x.split("_")[8]),
            # int(x.split("_")[9]),
            # float(x[0:-4].split("_")[10]),
            int(x.split("_")[4]),  # instance_num
            float(x.split("_")[6]),  # beta_value
            int(x.split("_")[7]),  # anneal_time
            float(x[0:-4].split("_")[8]),  # anneal_param
        )
        filenames = sorted(os.listdir(DATA), key=sort_key)
        for filename in filenames:

            file_path = os.path.join(DATA, filename)
            if os.path.isfile(file_path):
                print(f"optimizing {filename}")
                name = filename[0:-4]
                parameters = name.split("_")
                instance_num = parameters[4]
                beta_value = parameters[6]
                if int(instance_num) >= 6:
                    print(f"Skipping instance {instance_num}")
                    continue
                # if float(beta_value) <= 10:
                    # print(f"Skipping beta {beta_value} for instance {instance_num}")
                    # continue
                anneal_time = parameters[7]
                anneal_param = parameters[8]
                chain_length = "6"  # For P6 instances

                # Load instance from pegasus_native directory
                instance_file = os.path.join(
                    ROOT,
                    "data",
                    "instances",
                    "subpegasus_native",
                    f"P6_{inst_type}_{instance_num}.pkl",
                )
                with open(instance_file, "rb") as f:
                    inst_data = pickle.load(f)
                    # Handle both Instance namedtuple and list format
                    if hasattr(inst_data, "h") and hasattr(inst_data, "J"):
                        h = inst_data.h
                        J = inst_data.J
                    elif isinstance(inst_data, list) and len(inst_data) == 2:
                        h = inst_data[0]
                        J = inst_data[1]
                    else:
                        raise ValueError(f"Unknown instance format in {instance_file}")

                    h_vect, J_vect, h_new, J_new, key_map = vectorize_2d(h, J)

                df = pd.read_csv(file_path, index_col=0)

                configurations = []
                E_final = []
                Q_vect = []

                # Create BQM once outside the loop for efficiency
                bqm = BinaryQuadraticModel.from_ising(h_new, J_new)

                for row in df.itertuples():
                    state = eval(row.sample)
                    # Map the state using key_map if it exists, otherwise use state as-is
                    if key_map:
                        mapped_state = {
                            key_map[k]: v for k, v in state.items() if k in key_map
                        }
                    else:
                        mapped_state = state

                    # Convert to BQM energy calculation
                    energy_dict = bqm.energy(mapped_state)

                    # Convert state to sorted array for vectorized energy calculation
                    sorted_state = sorted(mapped_state.items(), key=lambda x: x[0])
                    state_array = np.array([v for k, v in sorted_state])
                    energy_np = energy(state_array, h_vect, J_vect)
                    assert (
                        abs(energy_dict - energy_np) < 1e-10
                    ), f"energy_dict: {energy_dict}, energy_np: {energy_np}"
                    configurations.append(state_array)

                    energy_final = float(row.energy)
                    assert (
                        abs(energy_final - energy_np) < 1e-10
                    ), f"energy_final: {energy_final}, energy_np: {energy_np}"
                    E_final.append(energy_final / int(chain_length))  # per spin

                    init_state = eval(row.init_state)
                    if key_map:
                        mapped_init_state = {
                            key_map[k]: v for k, v in init_state.items() if k in key_map
                        }
                    else:
                        mapped_init_state = init_state

                    sorted_init_state = sorted(
                        mapped_init_state.items(), key=lambda x: x[0]
                    )
                    init_state_array = np.array([v for k, v in sorted_init_state])
                    energy_init1 = bqm.energy(mapped_init_state)
                    energy_init2 = energy(init_state_array, h_vect, J_vect)
                    assert (
                        abs(energy_init1 - energy_init2) < 1e-10
                    ), f"energy_init1: {energy_init1}, energy_init2: {energy_init2}"
                    Q_vect.append(
                        (energy_final - energy_init1) / int(chain_length)
                    )  # per spin

                optim = optimize.minimize(
                    pseudo_likelihood_2d_vectorised,
                    np.array([1.0]),
                    args=(h_vect, J_vect, np.array(configurations)),
                )

                # Create key tuple for results storage
                result_key = (instance_num, beta_value, anneal_time, anneal_param)

                with open(os.path.join(RESULTS, f"betas2_P{chain_length}.pkl"), "wb") as f:
                    betas[result_key] = optim.x.item() * beta_units
                    pickle.dump(betas, f)
                print("result: beta = ", optim.x.item())
                with open(
                    os.path.join(RESULTS, f"energies_P{chain_length}.pkl"), "wb"
                ) as f2:
                    E_mean, E_var = np.mean(np.array(E_final)), np.var(np.array(E_final))
                    energies[result_key] = (E_mean * energy_units, E_var * energy_units**2)
                    pickle.dump(energies, f2)
                print("result: energies = ", (E_mean, E_var))
                with open(os.path.join(RESULTS, f"Q_P{chain_length}.pkl"), "wb") as f3:
                    Q_mean, Q_var = np.mean(np.array(Q_vect)), np.var(np.array(Q_vect))
                    Q[result_key] = (Q_mean * energy_units, Q_var * energy_units**2)
                    pickle.dump(Q, f3)
                print("result: Q = ", (Q_mean, Q_var))
                with open(os.path.join(RESULTS, f"Q_dist_P{chain_length}.pkl"), "wb") as f4:
                    Q_dist[result_key] = np.array(Q_vect)
                    pickle.dump(Q_dist, f4)


if __name__ == "__main__":
    main()
