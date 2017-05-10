from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import unittest
import numpy as np

from doppelganger import listbalancer


class ListBalancerTests(unittest.TestCase):

    def _mock_list_consistent(self):
        hh_table = np.mat([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1]
        ])
        A = np.mat([
            81.,
            101.,
            151.,
            429.,
            580.,
        ])
        w = np.matrix([[
            81.,
            101.,
            151.,
            429.
        ]])
        n_samples, n_controls = hh_table.shape

        expected_weights = np.matrix([
            [81.],
            [101.],
            [429.],
            [580.]
        ])

        return (hh_table, A, w, expected_weights)

    def _mock_list_inconsistent(self):
        hh_table = np.mat([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 2, 1],
            [0, 0, 1, 1, 2]
        ])
        A = np.mat([
            81.,
            101.,
            151.,
            429.,
            299.,
        ])
        w = np.matrix([
            [79.,
             99.,
             101.,
             49.]
        ])
        n_samples, n_controls = hh_table.shape
        mu = np.mat([1] * n_controls)

        expected_weights = np.matrix([
            [81.,
             101.,
             100.,
             51.]
        ])

        return (hh_table, A, w, mu, expected_weights)

    def _mock_list_infeasible_marginal(self):
        hh_table = np.mat([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ])
        A = np.mat([
            [0, 324.],
            [0, 357.],
            [0, 138.],
            [0, 183]
        ]).T
        w = np.matrix([
            [0., 0., 0., 0.],
            [79., 99., 101., 49.]
        ])
        n_samples, n_controls = hh_table.shape
        mu = np.mat(np.ones((2, n_controls)))

        expected_weights = np.matrix([
            [0., 0., 0., 0.],
            [340.45637778, 137.42776531, 122.81489579, 59.58339833]
        ])

        return (hh_table, A, w, mu, expected_weights)

    def _mock_hh_weights(self):
        hh_table = np.mat([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 2, 1],
            [0, 0, 1, 1, 2]
        ])
        hh_weights = np.mat([
            [80.79, 100.6, 100.7, 49.32],
            [80.79, 100.6, 100.7, 49.32],
        ])
        expected_hh_discretized = np.mat([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ])
        return hh_table, hh_weights, expected_hh_discretized

    def test_balance_cvx(self):
        hh_table, A, w, expected_weights = self._mock_list_consistent()
        n_samples, n_controls = hh_table.shape
        hh_weights = listbalancer.balance_cvx(hh_table, A, w)
        np.testing.assert_allclose(
            hh_weights, expected_weights, rtol=1, atol=0)

    def test_balance_multi_cvx(self):
        hh_table, A, w, mu, expected_weights = self._mock_list_inconsistent()

        # Extend the data
        n_tracts = 10
        A_extend = np.mat(np.tile(A, (n_tracts, 1)))
        w_extend = np.mat(np.tile(w, (n_tracts, 1)))
        mu_extend = np.mat(np.tile(mu, (n_tracts, 1)))
        B = np.mat(np.dot(np.ones((1, n_tracts)), A_extend)[0])
        expected_weights_extend = np.mat(
            np.tile(expected_weights, (n_tracts, 1)))
        gamma = 1000.
        meta_gamma = 1000.
        hh_weights, z, q = listbalancer.balance_multi_cvx(
            hh_table, A_extend, B, w_extend, gamma * mu_extend.T, meta_gamma
        )
        np.testing.assert_allclose(
            hh_weights, expected_weights_extend, rtol=1, atol=0)

    def test_balance_multi_trust_initial(self):
        # TODO: Set mock values so this is meaningful
        hh_table, A, w, mu, expected_weights = self._mock_list_inconsistent()
        B = np.mat(np.dot(np.ones((1, 1)), A)[0])
        gamma = 1.
        hh_weights, z, q = listbalancer.balance_multi_cvx(
            hh_table, A, B, w, gamma * mu.T
        )

        np.testing.assert_allclose(
            hh_weights, w, rtol=0.05, atol=0)

    def test_balance_multi_trust_controls(self):
        # TODO: Set mock values so this is meaningful
        hh_table, A, w, mu, expected_weights = self._mock_list_inconsistent()
        B = np.mat(np.dot(np.ones((1, 1)), A)[0])
        gamma = 10000.
        hh_weights, z, q = listbalancer.balance_multi_cvx(
            hh_table, A, B, w, gamma * mu.T
        )
        np.testing.assert_allclose(
            hh_weights, expected_weights, rtol=0.05, atol=0)

    def test_balance_multi_zero_marginal(self):
        hh_table, A, w, mu, expected_weights = \
            self._mock_list_infeasible_marginal()
        n_tracts = A.shape[0]
        B = np.mat(np.dot(np.ones((1, n_tracts)), A)[0])

        gamma = 10000.
        hh_weights, z, q = listbalancer.balance_multi_cvx(
            hh_table, A, B, w, gamma * mu.T
        )

        np.testing.assert_allclose(
            hh_weights, expected_weights, rtol=0.05, atol=0)

    def test_discretize_multi_weights(self):
        hh_table, hh_weights, expected_hh_discretized = self._mock_hh_weights()
        hh_discretized = listbalancer.discretize_multi_weights(
            hh_table, hh_weights)
        np.testing.assert_array_equal(
            hh_discretized, expected_hh_discretized)
