
import numpy as np
import dwave_networkx as dnx
import time
from src.utils import gibbs_sampling_efficient
from itertools import product
from dimod import BinaryQuadraticModel
from math import exp


rng = np.random.default_rng()


if __name__ == '__main__':

    g = dnx.pegasus_graph(16, nice_coordinates=True)
    h = {dnx.pegasus_coordinates(16).nice_to_linear(node): np.random.uniform(-1, 1) for node in g.nodes()}
    J = {(dnx.pegasus_coordinates(16).nice_to_linear(u), dnx.pegasus_coordinates(16).nice_to_linear(v)):
         np.random.uniform(-1, 1) for (u, v) in g.edges()}
    start = time.time()
    s = gibbs_sampling_efficient(h, J, 1.0, 10000)
    end = time.time()
    print(end - start)