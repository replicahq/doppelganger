# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import logging
import cvxpy as cvx
import numpy as np

logging.basicConfig(filename='logs', filemode='a', level=logging.INFO)


def _insert_append(arr, indices, values, axis=0):
    """Insert / Append values to array along given axis

    Args:
    arr (numpy array): Array to insert/append values
    indices (list(int)): indices before which values are inserted
    values (numpy array): Values to insert into arr
    axis (int): Axis along which to insert values

    Returns:
    numpy array: A copy of arr with values inserted and appended
    """
    insert_filter = indices < arr.shape[axis]
    insert_inds = indices[insert_filter]
    n_append = indices[~insert_filter].shape[0]
    append_shape = (1, n_append) if axis else (n_append, 1)
    arr_append = np.tile(values, append_shape)

    arr_update = np.insert(arr, insert_inds, values, axis=axis)
    arr_update = np.concatenate((arr_update, arr_append), axis=axis)
    return arr_update


def balance_cvx(hh_table, A, w, mu=None, verbose_solver=False):
    """Maximum Entropy allocaion method for a single unit

    Args:
        hh_table (numpy matrix): Table of households categorical data
        A (numpy matrix): Area marginals (controls)
        w (numpy array): Initial household allocation weights
        mu (numpy array): Importance weights of marginals fit accuracy
        verbose_solver (boolean): Provide detailed solver info

    Returns:
        (numpy matrix, numpy matrix): Household weights, relaxation factors
    """

    n_samples, n_controls = hh_table.shape
    x = cvx.Variable(n_samples)

    if mu is None:
        objective = cvx.Maximize(
            cvx.sum_entries(cvx.entr(x) + cvx.mul_elemwise(cvx.log(w.T), x))
        )

        constraints = [
            x >= 0,
            x.T * hh_table == A,
        ]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.SCS, verbose=verbose_solver)

        return x.value

    else:
        # With relaxation factors
        z = cvx.Variable(n_controls)

        objective = cvx.Maximize(
            cvx.sum_entries(cvx.entr(x) + cvx.mul_elemwise(cvx.log(w.T), x)) +
            cvx.sum_entries(mu * (cvx.entr(z)))
        )

        constraints = [
            x >= 0,
            z >= 0,
            x.T * hh_table == cvx.mul_elemwise(A, z.T),
        ]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.SCS, verbose=verbose_solver)

        return x.value, z.value


def balance_multi_cvx(hh_table, A, B, w, mu=1000., meta_mu=1000., verbose_solver=False):
    """Maximum Entropy allocation method for multiple balanced units

    Args:
        hh_table (numpy matrix): Table of households categorical data
        A (numpy matrix): Area marginals (controls)
        B (numpy matrix): Meta-marginals
        w (numpy array): Initial household allocation weights
        mu (float): Importance weights of marginals for accuracy of fit
        meta_mu (float): Importance weights of meta-marginals for accuracy of fit
        verbose_solver (boolean): Provide detailed solver info

    Returns:
        (numpy matrix, numpy matrix, numpy matrix): Household weights,
            relaxation factors, relaxation factors,
    """

    n_samples, n_controls = hh_table.shape

    # Solver won't converge with zero marginals. Identify and remove.
    zero_marginals = np.where(~A.any(axis=1))[0]
    zero_weights = np.zeros((1, n_samples))

    if zero_marginals.size:
        logging.info(
            '%i tract(s) with zero marginals encountered. '
            'Setting weights to zero'.format(zero_marginals.size)
        )

        # Need to remove problem tracts and add a row of zeros later
        A = np.delete(A, zero_marginals, axis=0)
        w = np.delete(w, zero_marginals, axis=0)
        mu = np.delete(mu, zero_marginals, axis=1)

    n_tracts = w.shape[0]
    x = cvx.Variable(n_tracts, n_samples)

    # Relative weights of tracts
    # (need to reshape for numpy broadcasting)
    wa = (np.sum(A, axis=1) / np.sum(A)).reshape(-1, 1)
    w_relative = (np.array(w) * np.array(wa))

    # With relaxation factors
    z = cvx.Variable(n_controls, n_tracts)
    q = cvx.Variable(n_controls)

    identity = np.ones((n_tracts, 1))

    solved = False
    while not solved:
        objective = cvx.Maximize(
            cvx.sum_entries(
                cvx.entr(x) + cvx.mul_elemwise(cvx.log(np.e * w_relative), x)
            ) +
            cvx.sum_entries(
                cvx.mul_elemwise(
                    mu, cvx.entr(z) + cvx.mul_elemwise(cvx.log(np.e), z)
                )
            ) +
            cvx.sum_entries(
                cvx.mul_elemwise(
                    meta_mu, cvx.entr(q) + cvx.mul_elemwise(cvx.log(np.e), q)
                )
            )
        )

        constraints = [
            x >= 0,
            z >= 0,
            q >= 0,
            x * hh_table == cvx.mul_elemwise(A, z.T),
            cvx.mul_elemwise(A.T, z) * identity == cvx.mul_elemwise(B.T, q)
        ]

        prob = cvx.Problem(objective, constraints)

        try:
            prob.solve(verbose=verbose_solver)
            solved = True

        except cvx.SolverError:
            if np.all(mu == 1):
                # We can't reduce mu any further
                break
            mu = np.where(mu > 10, mu - 10, 1)
            logging.info('Solver error encountered. Importance weights have been relaxed.')

    if not np.any(x.value):
        logging.exception('Solution infeasible. Using initial weights.')

    # If we didn't get a value return the initial weights
    weights_out = x.value if np.any(x.value) else w_relative

    # Insert zeros
    if zero_marginals.size:
        # Due to numpy insert behavior, we need to differentiate between the
        # values that go into the middle of the array, and the values that get appended
        weights_out = _insert_append(
            weights_out, zero_marginals, zero_weights, axis=0)

    return weights_out


def discretize_multi_weights(hh_table, x, gamma=100., verbose_solver=False):
    """Discretize weights in household table for multiple tracts

    Arguments:
        hh_table (numpy matrix): Table of households categorical data
        x (numpy matrix): Household weights
        gamma (float): Relaxation weight
        verbose_solver (boolean): Provide detailed solver info

    Returns:
        numpy array: Discretized household weights
    """

    n_samples, n_controls = hh_table.shape

    # Solver won't converge with zero weights. Identify and remove.
    zero_weights_inds = np.where(~x.any(axis=1))[0]
    zero_weights = np.zeros((1, n_samples))

    if zero_weights_inds.size:
        logging.info(
            '{} tract(s) with zero weight rows encountered. '
            'Setting weights to zero'.format(zero_weights_inds.size)
        )

        # Need to remove problem tracts and add a row of zeros later
        x = np.delete(x, zero_weights_inds, axis=0)

    n_tracts = x.shape[0]

    # Integerize x values
    x_int = x.astype(int)

    # Get residuals in new marginals from truncating to int
    A_residuals = np.dot(x, hh_table) - np.dot(x_int, hh_table)
    x_residuals = x - x_int

    # Coefficients in objective function
    x_log = np.log(x_residuals)

    # Decision variables for optimization
    y = cvx.Variable(n_tracts, n_samples)

    # Relaxation factors
    U = cvx.Variable(n_tracts, n_controls)
    V = cvx.Variable(n_tracts, n_controls)

    objective = cvx.Maximize(
        cvx.sum_entries(
            cvx.sum_entries(cvx.mul_elemwise(x_log, y), axis=1) -
            (gamma) * cvx.sum_entries(U, axis=1) -
            (gamma) * cvx.sum_entries(V, axis=1)
        )
    )

    constraints = [
        y * hh_table <= A_residuals + U,
        y * hh_table >= A_residuals - V,
        U >= 0,
        V >= 0,
        y >= 0,
        y <= 1.0
    ]

    prob = cvx.Problem(objective, constraints)

    try:
        prob.solve(verbose=verbose_solver)

    except cvx.SolverError:
        logging.exception(
            'Solver error encountered in weight discretization. Weights will be rounded.')

    weights_out = y.value if np.any(y.value) else x_residuals

    # Insert zeros
    if zero_weights_inds.size:
        weights_out = _insert_append(weights_out, zero_weights_inds, zero_weights, axis=0)

    # Make results binary and return
    return np.array(weights_out > 0.5).astype(int)
