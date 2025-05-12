import unittest
import numpy as np
import dwave_networkx as dnx
import time
from src.utils import gibbs_sampling_ising_vectorized_2d
from itertools import product
from dimod import BinaryQuadraticModel
from math import exp


rng = np.random.default_rng()


class GibbsSampling(unittest.TestCase):
    ...


if __name__ == '__main__':
    #unittest.main()
    g = dnx.pegasus_graph(16, nice_coordinates=True)
    g = dnx.pegasus_coordinates(16).nice_to_linear(g)
    h = {node: np.random.uniform(-1, 1) for node in g.nodes()}
    J = {edge: np.random.uniform(-1, 1) for edge in g.edges()}
    start = time.time()
    s = gibbs_sampling_ising_vectorized_2d(h, J, 1.0, 10000)
    end = time.time()
    print(end - start)