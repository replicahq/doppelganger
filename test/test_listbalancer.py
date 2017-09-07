# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

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
        _, n_controls = hh_table.shape

        expected_weights = np.matrix([
            [81.],
            [101.],
            [151.],
            [429.]
        ])

        mu = np.mat([1] * n_controls)

        return (hh_table, A, w, mu, expected_weights)

    def _mock_list_relaxed(self):
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
        _, n_controls = hh_table.shape

        expected_weights = np.matrix([
            [45.],
            [52.],
            [65.],
            [98.]
        ])

        mu = np.mat([1] * n_controls)

        return (hh_table, A, w, mu, expected_weights)

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
        _, n_controls = hh_table.shape
        mu = np.mat([1] * n_controls)

        expected_weights = np.matrix([
            [70.66,
             88.66,
             85.47,
             45.72]
        ])

        return (hh_table, A, w, mu, expected_weights)

    def _mock_list_infeasible(self):
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
            [1000.,
             0.,
             0.,
             0.]
        ])
        _, n_controls = hh_table.shape
        mu = np.mat([1] * n_controls)

        expected_weights = np.matrix([
            [100.,    0.,    0.,    0.]
        ])

        return (hh_table, A, w, mu, expected_weights)

    def _mock_list_infeasible_marginal(self):
        hh_table = np.mat([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ])
        A = np.mat([
            [0., 324., 0, 0.],
            [0., 357., 0, 0.],
            [0., 138., 0., 0.],
            [0., 183., 0., 0.],
        ]).T
        w = np.matrix([
            [0., 0., 0., 0., 0.],
            [79., 99., 101., 49., 200],
            [0., 0., 0., 0., 0],
            [0., 0., 0., 0., 0],
        ])
        _, n_controls = hh_table.shape
        mu = np.mat(np.ones((2, n_controls)))

        expected_weights = np.matrix([
            [0., 0., 0., 0., 0.],
            [340.46, 58.09, 69.56, 33.75, 80.83],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
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

    def _mock_hh_weights_zeroed(self):
        hh_table = np.mat([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 2, 1],
            [0, 0, 1, 1, 2]
        ])
        hh_weights = np.mat([
            [0., 0., 0., 0.],
            [80.79, 100.6, 100.7, 49.32],
            [0., 0., 0., 0.],
        ])
        expected_hh_discretized = np.mat([
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ])
        return hh_table, hh_weights, expected_hh_discretized

    def test_balance_cvx(self):
        hh_table, A, w, _, expected_weights = self._mock_list_consistent()
        hh_weights = listbalancer.balance_cvx(hh_table, A, w)
        np.testing.assert_allclose(
            hh_weights, expected_weights, rtol=0.01, atol=0)

    def test_balance_cvx_relaxed(self):
        hh_table, A, w, mu, expected_weights = self._mock_list_relaxed()
        hh_weights, _ = listbalancer.balance_cvx(hh_table, A, w, mu)
        np.testing.assert_allclose(
            hh_weights, expected_weights, rtol=0.01, atol=0)

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
        hh_weights = listbalancer.balance_multi_cvx(
            hh_table, A_extend, B, w_extend, gamma * mu_extend.T, meta_gamma)
        np.testing.assert_allclose(
            hh_weights, expected_weights_extend, rtol=0.01, atol=0)

    def test_balance_multi_cvx_infeasible(self):
        hh_table, A, w, mu, expected_weights = self._mock_list_infeasible()

        # Extend the data
        n_tracts = 10
        A_extend = np.mat(np.tile(A, (n_tracts, 1)))
        w_extend = np.mat(np.tile(w, (n_tracts, 1)))
        expected_weights_extend = np.mat(np.tile(expected_weights, (n_tracts, 1)))
        mu_extend = np.mat(np.tile(mu, (n_tracts, 1)))
        B = np.mat(np.dot(np.ones((1, n_tracts)), A_extend)[0])
        gamma = 10.
        meta_gamma = 1000.
        hh_weights = listbalancer.balance_multi_cvx(
            hh_table, A_extend, B, w_extend, gamma * mu_extend.T, meta_gamma
        )
        np.testing.assert_allclose(
            hh_weights, expected_weights_extend, rtol=0.01, atol=0)

    def test_balance_multi_trust_initial(self):
        hh_table, A, w, mu, _ = self._mock_list_inconsistent()
        B = np.mat(np.dot(np.ones((1, 1)), A)[0])
        gamma = 1.
        hh_weights = listbalancer.balance_multi_cvx(
            hh_table, A, B, w, gamma * mu.T
        )
        np.testing.assert_allclose(
            hh_weights, w, rtol=0.05, atol=0)

    def test_balance_multi_trust_controls(self):
        hh_table, A, w, mu, expected_weights = self._mock_list_consistent()
        B = np.mat(np.dot(np.ones((1, 1)), A)[0])
        gamma = 100000.
        hh_weights = listbalancer.balance_multi_cvx(
            hh_table, A, B, w, gamma * mu.T
        )
        np.testing.assert_allclose(
            hh_weights, expected_weights.T, rtol=0.05, atol=0)

    def test_balance_multi_zero_marginal(self):
        hh_table, A, w, mu, expected_weights = \
            self._mock_list_infeasible_marginal()
        n_tracts = A.shape[0]
        B = np.mat(np.dot(np.ones((1, n_tracts)), A)[0])
        gamma = 10000.
        hh_weights = listbalancer.balance_multi_cvx(
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

    def test_discretize_multi_zero_weights(self):
        hh_table, hh_weights, expected_hh_discretized = self._mock_hh_weights_zeroed()
        hh_discretized = listbalancer.discretize_multi_weights(
            hh_table, hh_weights)
        np.testing.assert_array_equal(
            hh_discretized, expected_hh_discretized)
