# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import unittest
import pandas

from doppelganger import datasource, inputs, Preprocessor


class DataSourceTest(unittest.TestCase):

    def _mock_dirty_household_input(self):
        return [
            {'serialno': 'a', 'np': '1', 'veh': '1', 'fincp': ''},
            {'serialno': 'b', 'np': '2', 'veh': '6', 'fincp': '0070000'},
            {'serialno': 'c', 'np': '1', 'veh': '', 'fincp': '0000000'}
        ]

    def _mock_dirty_people_input(self, weight=1):
        def mock_person(serialno, age, sex, income):
            return {
                'serialno': serialno,
                'agep': age,
                'sex': sex,
                'wagp': income,
                'pwgtp': weight
            }
        return [
            mock_person('a', '15', '1', ''),
            mock_person('b', '50', '2', '0070000'),
            mock_person('b', '70', '1', '0015000'),
            mock_person('c', '25', '1', '0000000')
        ]

    def _mock_dirty_household_puma_state_input(self):
        return [
            {'serialno': 'a', 'puma': '00106', 'st': '06'},
            {'serialno': 'b', 'puma': '00107', 'st': '06'},
            {'serialno': 'c', 'puma': '00106', 'st': '07'}
        ]

    def test_clean_data(self):
        pums_data = datasource.PumsData(
            pandas.DataFrame(self._mock_dirty_household_input())
        )
        cleaned = pums_data.clean([
            inputs.SERIAL_NUMBER.name,
            inputs.NUM_PEOPLE.name,
            inputs.NUM_VEHICLES.name,
            inputs.HOUSEHOLD_INCOME.name
        ],
            Preprocessor())
        actual = cleaned.data.loc[1].to_dict()
        expected = {
            inputs.SERIAL_NUMBER.name: 'b',
            inputs.NUM_PEOPLE.name: '2',
            inputs.NUM_VEHICLES.name: '3+',
            inputs.HOUSEHOLD_INCOME.name: '40000+'
        }
        self.assertDictEqual(actual, expected)

    def test_clean_data_one_field(self):
        pums_data = datasource.PumsData(
            pandas.DataFrame(self._mock_dirty_household_input())
        )
        cleaned = pums_data.clean([inputs.NUM_PEOPLE.name], Preprocessor())
        actual = cleaned.data.loc[1].to_dict()
        expected = {
            inputs.NUM_PEOPLE.name: '2',
        }
        self.assertDictEqual(actual, expected)

    def test_clean_data_filter_length(self):
        pums_data = datasource.PumsData(
            pandas.DataFrame(self._mock_dirty_household_puma_state_input())
        )
        field_names = [inputs.SERIAL_NUMBER.name, inputs.STATE.name, inputs.PUMA.name]
        cleaned = pums_data.clean(field_names, Preprocessor())
        cleaned_state = pums_data.clean(field_names, Preprocessor(), state='06')
        cleaned_puma = pums_data.clean(field_names, Preprocessor(), puma='00106')
        cleaned_both = pums_data.clean(field_names, Preprocessor(), state='06', puma='00106')
        self.assertEqual(len(cleaned.data), 3)
        self.assertEqual(len(cleaned_state.data), 2)
        self.assertEqual(len(cleaned_puma.data), 2)
        self.assertEqual(len(cleaned_both.data), 1)
