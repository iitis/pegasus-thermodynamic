import unittest
import numpy as np
import dwave_networkx as dnx

from src.utils import pseudo_likelihood, extend, pseudo_likelihood_2d_vectorised, vectorize


class TestPseudoLikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = dnx.pegasus_graph(4, nice_coordinates=True)

    def test_equal_vectorized(self):
        N = 300
        D = 100

        J = {(i, i+1): np.random.uniform(-1, 1) for i in range(N-1)}
        J = extend(J)
        h = {i: np.random.uniform(-1, 1) for i in range(N)}

        h_vect, J_vect = vectorize(h, J)
        samples = np.random.choice([-1, 1], size=(D, N))
        beta = 1.0

        a = pseudo_likelihood(beta, h, J, samples)

        b = pseudo_likelihood_2d_vectorised(beta, h_vect, J_vect, samples)
        self.assertAlmostEqual(a, b)


if __name__ == '__main__':
    unittest.main()
