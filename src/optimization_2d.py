import os
import pickle

import numpy as np
import pandas as pd

from scipy import optimize
from utils import pseudo_likelihood_2d, extend, vectorize, energy, vectorize_2d
from collections import namedtuple
from dimod import BinaryQuadraticModel
Instance = namedtuple("Instance", ["h", "J", "name"])

PHYSICAL_UNITS = False

B0 = 1.0 # actually B0 = 8.58.. but this is included into annealing schedule
h = 6.62607015e-34  # Planck constant, in J/Hz
kb = 1.380649e-23  # Boltzmann constant, in J/K
energy_units = (B0/2) * 10**9 * h
beta_units = kb / energy_units
if PHYSICAL_UNITS == False:
    energy_units = 1
    beta_units = 1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()
# DATA = os.path.join(ROOT, "data", "raw_data", "scalability_2d_cbfm")
# RESULTS = os.path.join(ROOT, "data", "results", "scalability_2d_cbfm")

itype = "const"  

DATA = os.path.join(ROOT, "data", "raw_data", f"annealing_param_2d_{itype}")
RESULTS = os.path.join(ROOT, "data", "results", f"annealing_param_2d_{itype}")

if not os.path.exists(RESULTS):
  os.makedirs(RESULTS)

betas = {}
energies = {}
Q = {}
Q_dist = {}

# with open(os.path.join(ROOT, "data", "instance.pkl"), "rb") as f:
#     h, J = pickle.load(f)
#     h_vect, J_vect = vectorize(h, J)
#     J = extend(J)
#     chain_length = len(h)

def main():
    sort_key = lambda x: int(x.split("_")[3])
    filenames = sorted(os.listdir(DATA), key=sort_key)
    for filename in filenames:
        
        file_path = os.path.join(DATA, filename)
        if os.path.isfile(file_path):
            print(f"optimizing {filename}, {itype}")
            name = filename[0:-4]
            parameters = name.split("_")
            chain_length = parameters[3]
            anneal_time = parameters[4]
            anneal_param = parameters[5]
            # if int(chain_length) != 500:
            #     continue
            # with open(os.path.join(ROOT, "data", f"instance_{chain_length}_large_h.pkl"), "rb") as f:
            with open(os.path.join(ROOT, "data", "instances", f"p{chain_length}_{itype}.pkl"), "rb") as f:
                # h, J = pickle.load(f)
                inst = pickle.load(f)
                h = inst.h
                J = inst.J
                h_vect, J_vect, h, J, key_map = vectorize_2d(h, J)
                J_ext = extend(J)

            df = pd.read_csv(file_path, index_col=0)

            configurations = []
            E_final = []
            Q_vect = []
            for row in df.itertuples():
                state = eval(row.sample)
                state = list(state.values())
                configurations.append(state)
                # configurations.append(list(state.values()))
                energy_final = float(row.energy)
                E_final.append(energy_final / int(chain_length))  # per spin
                init_state = eval(row.init_state)
                init_state = np.array(list(init_state.values()))
                # bqm = BinaryQuadraticModel.from_ising(h, J)
                # energy_init = bqm.energy(init_state)
                energy_init = energy(init_state, h_vect, J_vect)
                Q_vect.append((energy_final - energy_init) / int(chain_length))  # per spin

            optim = optimize.minimize(pseudo_likelihood_2d, np.array([1.0]), args=(h, J_ext, np.array(configurations)))
            with open(os.path.join(RESULTS, f"betas2_P{chain_length}.pkl"), "wb") as f:
                betas[(anneal_time, anneal_param)] = optim.x.item() * beta_units
                pickle.dump(betas, f)
            print("result: beta = ", optim.x.item())
            with open(os.path.join(RESULTS, f"energies_P{chain_length}.pkl"), "wb") as f2:
                E_mean, E_var = np.mean(np.array(E_final)), np.var(np.array(E_final))
                energies[(anneal_time, anneal_param)] = (E_mean * energy_units, E_var * energy_units**2)
                pickle.dump(energies, f2)
            print("result: energies = ", (E_mean, E_var))
            with open(os.path.join(RESULTS, f"Q_P{chain_length}.pkl"), "wb") as f3:
                Q_mean, Q_var = np.mean(np.array(Q_vect)), np.var(np.array(Q_vect))
                Q[(anneal_time, anneal_param)] = (Q_mean * energy_units, Q_var * energy_units**2)
                pickle.dump(Q, f3)
            print("result: Q = ", (Q_mean, Q_var))
            with open(os.path.join(RESULTS, f"Q_dist_P{chain_length}.pkl"), "wb") as f4:
                Q_dist[(anneal_time, anneal_param)] = np.array(Q_vect)
                pickle.dump(Q_dist, f4)


if __name__ == '__main__':
    main()
