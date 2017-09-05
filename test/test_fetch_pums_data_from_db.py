from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import unittest
import mock
import os
from doppelganger.scripts.fetch_pums_data_from_db import link_fields_to_inputs, fetch_pums_data
from doppelganger import inputs, config, allocation


class TestFetchPumsDataFromDB(unittest.TestCase):

    def _mock_config(self):
        conf = {
            "person_fields": [
                "age",
                "sex",
                "individual_income"
            ],
            "household_fields": [
                "num_people",
                "household_income",
                "num_vehicles"
            ],
            "preprocessing": {
                "individual_income": {
                    "bins": [
                        0,
                        20000,
                        40000,
                        80000,
                        100000
                    ]
                }
            },
            "network_config_files": {
                "person": os.path.dirname(__file__)
                + "/../examples/sample_data/sample_person_bn.json",
                "household": os.path.dirname(__file__)
                + "/../examples/sample_data/sample_household_bn.json"
            },
            "version": "0"
        }
        return config.Configuration.from_json(conf)

    _mock_params = {
                    'state_id': '01',
                    'puma_id': '00001',
                    'output_dir': '.',
                    'db_host': 'host1',
                    'db_database': 'database1',
                    'db_schema': 'schema1',
                    'db_table': 'persons',  # Hard-Coded for now
                    'db_user': 'user1',
                    'db_password': 'password1',
                    'census_api_key': 'census_key1',
                    'puma_tract_mappings': 'puma_tract_mappings',
                }

    def _mock_legit_person_input_list(self):
        return ['age', 'sex', 'individual_income']

    def _mock_illegit_person_input_list(self):
        return ['not_a_field', 'sex', 'individual_income']

    def test_link_fields_to_inputs(self):
        self.assertEqual(set([inputs.AGE, inputs.SEX, inputs.INDIVIDUAL_INCOME]),
                         link_fields_to_inputs(self._mock_legit_person_input_list()))
        self.assertRaises(ValueError, link_fields_to_inputs, self._mock_illegit_person_input_list())

    @mock.patch('doppelganger.scripts.fetch_pums_data_from_db.psycopg2')
    @mock.patch('doppelganger.scripts.fetch_pums_data_from_db.datasource.PumsData')
    def test_fetch_pums_data(self, mock_PumsData, mock_psycopg2):
        mock_config = self._mock_config()
        mock_db_connection = mock.Mock()
        mock_psycopg2.connect.return_value = mock_db_connection

        default_person_fields = allocation.DEFAULT_PERSON_FIELDS

        # local_person_fields = link_fields_to_inputs(mock_config.person_fields)
        # person_fields = default_person_fields.union(local_person_fields)

        with mock.patch(
                'doppelganger.scripts.fetch_pums_data_from_db.allocation'
                ) as mock_allocation:
            mock_allocation.DEFAULT_PERSON_FIELDS = \
                mock.PropertyMock(default_person_fields)
            persons_data, households_data = fetch_pums_data(
                            state_id=self._mock_params['state_id'],
                            puma_id=self._mock_params['puma_id'],
                            configuration=mock_config,
                            db_host=self._mock_params['db_host'],
                            db_database=self._mock_params['db_database'],
                            db_schema=self._mock_params['db_schema'],
                            db_user=self._mock_params['db_user'],
                            db_password=self._mock_params['db_password']
                        )
        mock_psycopg2.connect.assert_called()
        mock_PumsData.from_database.assert_called()
