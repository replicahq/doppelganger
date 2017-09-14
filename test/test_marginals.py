# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import unittest
from mock import patch

from doppelganger import Marginals


class MarginalsTest(unittest.TestCase):

    def _mock_marginals_file(self):
        return [{
            'STATEFP': '06',
            'COUNTYFP': '075',
            'PUMA5CE': '07507',
            'TRACTCE': '023001'
        }]

    def _mock_response(self):
        return dict(
            zip(['B11016_010E', 'B11016_003E', 'B11016_011E', 'B11016_012E',
                 'B11016_004E', 'B11016_005E', 'B11016_006E', 'B11016_007E',
                 'B11016_008E', 'B11016_013E', 'B11016_014E', 'B11016_015E',
                 'B11016_016E', 'B11016_010E', 'B11016_012E', 'B11016_004E',
                 'B11016_003E', 'B11016_011E', 'B11016_005E', 'B11016_006E',
                 'B11016_007E', 'B11016_008E', 'B11016_013E', 'B11016_014E',
                 'B11016_015E', 'B11016_016E',
                 'B01001_003E', 'B01001_004E', 'B01001_005E', 'B01001_006E',
                 'B01001_027E', 'B01001_028E', 'B01001_029E', 'B01001_030E',
                 'B01001_007E', 'B01001_008E', 'B01001_009E', 'B01001_010E',
                 'B01001_011E', 'B01001_012E', 'B01001_031E', 'B01001_032E',
                 'B01001_033E', 'B01001_034E', 'B01001_035E', 'B01001_036E',
                 'B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E',
                 'B01001_024E', 'B01001_025E', 'B01001_044E', 'B01001_045E',
                 'B01001_046E', 'B01001_047E', 'B01001_048E', 'B01001_049E',
                 'B01001_013E', 'B01001_014E', 'B01001_015E', 'B01001_016E',
                 'B01001_017E', 'B01001_018E', 'B01001_019E', 'B01001_037E',
                 'B01001_038E', 'B01001_039E', 'B01001_040E', 'B01001_041E',
                 'B01001_042E', 'B01001_043E',
                 'B08141_002E', 'B08141_003E', 'B08141_004E', 'B08141_005E',
                 'state', 'county', 'tract'],
                ['168', '267', '74', '0', '304', '240', '91', '52', '122',
                 '17', '0', '0', '0', '168', '0', '304', '267', '74', '240',
                 '91', '52', '122', '17', '0', '0', '0',
                 '126', '100', '123', '63', '160', '46', '135', '156', '40',
                 '44', '25', '68', '117', '241', '55', '49', '15', '68', '208',
                 '194', '94', '130', '78', '36', '29', '47', '42', '60', '71',
                 '32', '60', '34', '214', '135', '168', '109', '187', '133',
                 '149', '159', '175', '222', '171', '263', '73', '176',
                 '0', '1', '2', '3',
                 '06', '075', '023001']))

    def test_fetch_marginals(self):
        state = self._mock_marginals_file()[0]['STATEFP']
        puma = self._mock_marginals_file()[0]['PUMA5CE']
        with patch('doppelganger.marginals.Marginals._fetch_from_census',
                   return_value=self._mock_response()):
            marg = Marginals.from_census_data(
                    puma_tract_mappings=self._mock_marginals_file(), census_key=None,
                    state=state, pumas=set([puma])
                )
        expected = {
            'STATEFP': '06',
            'COUNTYFP': '075',
            'PUMA5CE': '07507',
            'TRACTCE': '023001',
            'age_0-17': '909',
            'age_18-34': '1124',
            'age_65+': '713',
            'age_35-64': '2334',
            'num_people_count': '1335',
            'num_people_1': '168',
            'num_people_3': '304',
            'num_people_2': '341',
            'num_people_4+': '522',
            'num_vehicles_0': '0',
            'num_vehicles_1': '1',
            'num_vehicles_2': '2',
            'num_vehicles_3+': '3'
        }
        result = marg.data.loc[0].to_dict()
        self.assertDictEqual(result, expected)

    def _mock_marginals_csv():
        return [{
            'STATEFP': '06',
            'COUNTYFP': '075',
            'PUMA5CE': '1',
            'TRACTCE': '100',
            'num_people_1': '40',
            'num_people_2+': '50',
        },
            {
            'STATEFP': '06',
            'COUNTYFP': '075',
            'PUMA5CE': '2',
            'TRACTCE': '101',
            'num_people_1': '80',
            'num_people_2+': '90',
        }]
