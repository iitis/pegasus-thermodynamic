import os
import pickle

import numpy as np
import pandas as pd

from scipy import optimize
from utils import pseudo_likelihood, extend, vectorize, energy
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
hrange = 2.0
hname = "large"
# RESULTS = os.path.join(ROOT, "data", "results", "paper", f"annealing_param_1d_pegasus_unif_{hname}_h")
# DATA = os.path.join(ROOT, "data", "raw_data", f"annealing_param_{hname}_h")
RESULTS = os.path.join(ROOT, "data", "results", "paper", f"new_scalability_1d_pegasus_unif_{hname}_h")


if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)

betas = {}
energies = {}
Q = {}
Q_dist = {}


def main():
  for L in [50, 100, 300, 1000, 2000, 3000]:
    DATA = os.path.join(ROOT, "data", "raw_data", f"scalability_pegasus_1d_chain_L_{L}_unif_{hname}_h_advantage_6.4")

    for filename in os.listdir(DATA):
        file_path = os.path.join(DATA, filename)
        if os.path.isfile(file_path):
            print(f"optimizing {filename}")
            name = filename[0:-4]
            parameters = name.split("_")
            chain_length = parameters[3]
            anneal_time = parameters[4]
            anneal_param = parameters[5]
            beta = parameters[6]
            # if int(chain_length) != 500:
            #     continue

            with open(
                os.path.join(ROOT, "data", "instances", "1d_pegasus_chains", f"pegasus_1D_chain_L_{chain_length}_unif_{hname}_h.pkl"),
                "rb",
            ) as f:
                h, J = pickle.load(f)
                h_vect, J_vect = vectorize(h, J)
                J_ext = extend(J)

            df = pd.read_csv(file_path, index_col=0)

            configurations = []
            E_final = []
            Q_vect = []
            for row in df.itertuples():
                if row.energy == "energy":
                    continue
                state = eval(row.sample)
                state = list(state.values())
                configurations.append(state)
                energy_final = float(row.energy)
                E_final.append(energy_final / int(chain_length))  # per spin
                init_state = eval(row.init_state)
                init_state = np.array(list(init_state.values()))
                energy_init = energy(init_state, h_vect, J_vect)
                Q_vect.append(
                    (energy_final - energy_init) / int(chain_length)
                )  # per spin

            optim = optimize.minimize(
                pseudo_likelihood,
                np.array([1.0]),
                args=(h, J_ext, np.array(configurations)),
            )
            with open(os.path.join(RESULTS, f"betas2_{chain_length}.pkl"), "wb") as f:
                betas[(chain_length, anneal_time, anneal_param, beta)] = optim.x.item() * beta_units
                pickle.dump(betas, f)
            print("result: beta = ", optim.x.item())
            with open(
                os.path.join(RESULTS, f"energies_{chain_length}.pkl"), "wb"
            ) as f2:
                E_mean, E_var = np.mean(np.array(E_final)), np.var(np.array(E_final))
                energies[(chain_length, anneal_time, anneal_param, beta)] = (
                    E_mean * energy_units,
                    E_var * energy_units**2,
                )
                pickle.dump(energies, f2)
            print("result: energies = ", (E_mean, E_var))
            with open(os.path.join(RESULTS, f"Q_{chain_length}.pkl"), "wb") as f3:
                Q_mean, Q_var = np.mean(np.array(Q_vect)), np.var(np.array(Q_vect))
                Q[(chain_length, anneal_time, anneal_param, beta)] = (
                    Q_mean * energy_units,
                    Q_var * energy_units**2,
                )
                pickle.dump(Q, f3)
            print("result: Q = ", (Q_mean, Q_var))
            with open(os.path.join(RESULTS, f"Q_dist_{chain_length}.pkl"), "wb") as f4:
                Q_dist[(chain_length, anneal_time, anneal_param, beta)] = np.array(Q_vect)
                pickle.dump(Q_dist, f4)


if __name__ == "__main__":
    main()
