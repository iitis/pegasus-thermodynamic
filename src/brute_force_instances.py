import h5py
import os
import heapq

import numpy as np
import pandas as pd

from src.utils import h5_tree
from math import inf
from copy import deepcopy
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from dimod import BinaryQuadraticModel, SampleSet
from dwave.samplers import SimulatedAnnealingSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng()


def read_instance(path: str):
    with h5py.File(path) as f:
        I = f["Ising"]["J_coo"]["I"][:]
        J = f["Ising"]["J_coo"]["J"][:]
        V = f["Ising"]["J_coo"]["V"][:]
        biases = f["Ising"]["biases"][:]
    return I, J, V, biases


def read_instance_spinglass(path: str):
    df = pd.read_csv(path, delimiter=" ", header=None, comment="#", names=["i", "j", "v"])
    print(df)
    n = df["j"].max()
    matrix = np.zeros((n, n))
    bias = np.zeros(n)
    for row in df.itertuples():
        if row.i == row.j:
            bias[row.i - 1] = row.v
        else:
            matrix[row.i - 1, row.j - 1] = row.v
    return matrix, bias

def create_dense_matrix(I, J, V, triu: bool = True) -> np.ndarray:
    n = max(I)
    matrix = np.zeros((n, n))
    for idx in range(len(I)):
        i = I[idx] - 1
        j = J[idx] - 1
        v = V[idx]
        if i > j and triu:
            i, j = j, i
        matrix[i, j] += v
    return matrix


def add_spectrum_to_data(old_file_path: str, new_file_path: str, highest_energies: list, lowest_energies: list,
                         name: str):

    I, J, V, biases = read_instance(old_file_path)

    with h5py.File(os.path.join(new_file_path, name), "w") as f:
        ising = f.create_group("Ising")
        biases_h5 = ising.create_dataset("biases", data=biases, dtype="float64",
                                         compression="gzip", compression_opts=9)
        j_coo = ising.create_group("J_coo")
        i_h5 = j_coo.create_dataset("I", data=I, dtype="i", compression="gzip", compression_opts=9)
        j_h5 = j_coo.create_dataset("J", data=J, dtype="i", compression="gzip", compression_opts=9)
        v_h5 = j_coo.create_dataset("V", data=V, dtype="float64",
                                    compression="gzip", compression_opts=9)

        spectrum = f.create_group("Spectrum")
        lowest = spectrum.create_group("Lowest")
        highest = spectrum.create_group("Highest")

        energies_lowest = lowest.create_dataset("energies", dtype="float64", compression="gzip", compression_opts=9)
        states_lowest = lowest.create_dataset("states", dtype="i", compression="gzip", compression_opts=9)

        energies_highest = highest.create_dataset("energies", dtype="float64", compression="gzip", compression_opts=9)
        states_highest = highest.create_dataset("states", dtype="i", compression="gzip", compression_opts=9)


def save_sol_only(save_path: str, name: str, matrix: np.ndarray, bias: np.ndarray, solution: list):

    states = []
    energies = []
    for energy, state in solution:
        states.append(list(state))
        energies.append(float(energy))

    states = np.array(states)
    energies = np.array(energies)
    with h5py.File(os.path.join(save_path, name), "w") as f:
        ising = f.create_group("Ising")
        biases_h5 = ising.create_dataset("biases", data=bias, dtype="float64",
                                         compression="gzip", compression_opts=9)
        J_h5 = ising.create_dataset("J", data=matrix, dtype="float64",
                                    compression="gzip", compression_opts=9)
        spectrum = f.create_group("Spectrum")
        states = spectrum.create_dataset("states", data=states, dtype="i", compression="gzip", compression_opts=9)
        energies = spectrum.create_dataset("energies", data=energies, dtype="float64", compression="gzip",
                                           compression_opts=9)


def ising_energy(state: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> float:
    spins = np.array(state)
    return np.dot(spins, np.dot(matrix, spins)) + np.dot(spins, bias)


def random_state(n: int):
    return np.array([rng.choice([-1, 1]) for _ in range(n)])


def process_state(state: list, matrix: np.ndarray, bias: np.ndarray):
    en = ising_energy(state, matrix, bias)
    return en, state


def process_spin_wrapper(args):
    return process_state(*args)


def get_solution_sim_anneal(path: str) -> SampleSet:
    I, J, V, biases = read_instance(path)
    matrix = create_dense_matrix(I, J, V)
    bqm = BinaryQuadraticModel(biases, matrix, vartype="SPIN")
    solution = SimulatedAnnealingSampler().sample(bqm, num_reads=1000)
    solution = solution.aggregate()

    return solution


def bruteforce(matrix: np.ndarray, bias: np.ndarray):
    n = len(bias)

    # Generate all possible spin combinations
    all_states = list(product([-1, 1], repeat=n))
    args = [(spin, matrix, bias) for spin in all_states]

    # Parallel processing pool
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_spin_wrapper, args),
            total=len(all_states),
            desc="Calculating bruteforce"
        ))

    scale = 1000
    ordered_results = list(sorted(results, key=lambda x: x[0]))

    # lowest_energies = deepcopy(ordered_results[-1:-scale])
    # highest_energies = deepcopy(ordered_results[0:scale-1])

    # return lowest_energies, highest_energies
    return ordered_results[0:100]

if __name__ == '__main__':
    path = os.path.join(ROOT, "data", "instances", "P2", "P2_07120.hdf5")
    path2 = os.path.join(ROOT, "data", "test_spinglasspeps")
    matrix, bias = read_instance_spinglass(os.path.join(path2, "cross_2_4_mdd.txt"))
    print(matrix)
    print(bias)
    sol = bruteforce(matrix, bias)
    print(sol)
    bqm = BinaryQuadraticModel(bias, matrix, vartype="SPIN")
    solution = SimulatedAnnealingSampler().sample(bqm, num_reads=1000)
    solution = solution.aggregate()
    print(solution)
    save_sol_only(path2, "cross_2_4_mdd.hdf5", matrix, bias, sol)