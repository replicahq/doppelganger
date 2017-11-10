# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import mock
from mock import Mock
import unittest
import pandas as pd
import numpy as np

from doppelganger import Accuracy
from doppelganger.accuracy import ErrorStat


class TestAccuracy(unittest.TestCase):
    def _mock_variable_bins(self):
        return [
                    ('num_people', '1'),
                    ('num_people', '3'),
                    ('num_people', '2'),
                    ('num_people', '4+'),
                    ('num_vehicles', '1'),
                    ('num_vehicles', '0'),
                    ('num_vehicles', '2'),
                    ('num_vehicles', '3+'),
                    ('age', '0-17'),
                    ('age', '18-34'),
                    ('age', '65+'),
                    ('age', '35-64'),
                ]

    def _mock_state_puma(self):
        return [('20', '00500'), ('20', '00602'), ('20', '00604'), ('29', '00901'), ('29', '00902')]

    def _mock_comparison_dataframe(self):
        # Just the top 10 lines of a sample PUMS file, counts will NOT line up with marginals.
        return pd.DataFrame(
                data=[
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    ],
                columns=['pums', 'marginal', 'gen'],
                index=self._mock_variable_bins())

    @mock.patch('doppelganger.Accuracy._comparison_dataframe')
    def test_error_metrics(self, mock_comparison_dataframe):
        accuracy = Accuracy(Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        accuracy.comparison_dataframe = self._mock_comparison_dataframe()
        self.assertEqual(accuracy.root_mean_squared_error(), (1.0, 1.0))
        self.assertListEqual(accuracy.root_squared_error().mean().tolist(), [1.0, 1.0])
        self.assertListEqual(accuracy.absolute_pct_error().mean().tolist(),
                             [2.0, 0.66666666666666663])

    @mock.patch('doppelganger.Accuracy.from_data_dir')
    @mock.patch('doppelganger.Accuracy._comparison_dataframe')
    def test_error_report(self, mock_comparison_datframe, mock_from_data_dir):
        accuracy = Accuracy(Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        accuracy.comparison_dataframe = self._mock_comparison_dataframe()
        accuracy.from_data_dir.return_value = accuracy

        state_puma = dict()
        state_puma['20'] = ['00500', '00602', '00604']
        state_puma['29'] = ['00901', '00902']

        expected_columns = ['marginal-pums', 'marginal-doppelganger']

        df_puma, df_variable, df_total =\
            accuracy.error_report(
                    state_puma, 'fake_dir',
                    marginal_variables=['num_people', 'num_vehicles', 'age'],
                    statistic=ErrorStat.ABSOLUTE_PCT_ERROR
                    )

        # Test df_total
        df_total_expected = pd.Series(
                [2.00000, 0.666667],
                index=expected_columns
                )
        self.assertTrue(all((df_total - df_total_expected) < 1))

        # Test df_puma
        expected_puma_data = np.reshape([2.0, 2/3.0]*5, (5, 2))
        df_expected_puma = pd.DataFrame(
                data=expected_puma_data,
                index=self._mock_state_puma(),
                columns=expected_columns
                )
        self.assertTrue((df_expected_puma == df_puma).all().all())

        # Test df_variable
        expected_variable_data = np.reshape([2.0, 2/3.0]*12, (12, 2))
        df_expected_variable = pd.DataFrame(
                data=expected_variable_data,
                index=self._mock_variable_bins(),
                columns=expected_columns
                )
        self.assertTrue((df_expected_variable == df_variable).all().all())

        # Test unimplemented statistic name
        try:
            self.assertRaises(
                Exception,
                Accuracy.error_report(
                        state_puma, 'fake_dir',
                        marginal_variables=['num_people', 'num_vehicles', 'age'],
                        statistic='wrong-statistic-name'
                        )
                )
        except Exception:
            pass
