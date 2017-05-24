# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import unittest


import os.path
from doppelganger import (
    allocation,
    Configuration,
)

CURRENT_VERSION = '0'


class TestConfig(unittest.TestCase):

    def _mock_config_files(self):
        return {
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

    def _mock_household_bn(self):
        return {
            "type": "household",
            "nodes": [
                "num_people",
                "household_income",
                "num_vehicles"
            ],
            "edges": {
                "num_people": [
                    "household_income",
                    "num_vehicles"
                ],
                "household_income": [
                    "num_vehicles"
                ]
            }
        }

    def _mock_person_bn(self):
        return {
            "type": "person",
            "nodes": [
                "age",
                "sex",
                "income"
            ],
            "edges": {
                "age": [
                    "income"
                ],
                "sex": [
                    "income"
                ]
            }
        }

    def test_from_json_version_empty(self):
        config_json = self._mock_config_files()
        del config_json['version']
        self.assertRaises(AssertionError, Configuration.from_json, config_json)

    def test_from_json_version_wrong(self):
        config_json = self._mock_config_files()
        config_json['version'] = '1'
        self.assertRaises(AssertionError, Configuration.from_json, config_json)

    def test_from_json_version_correct(self):
        config_json = self._mock_config_files()
        config = Configuration.from_json(config_json)
        self.assertEqual(config.version, CURRENT_VERSION)

    def test_get_all_person_fields(self):
        config_json = self._mock_config_files()
        config = Configuration.from_json(config_json)
        combined_set = set(config.get_all_person_fields())
        correct_set = set(tuple(set(
            field.name for field in allocation.DEFAULT_PERSON_FIELDS).union(
                set(config.person_fields))))
        self.assertEqual(combined_set, correct_set)

    def test_get_all_household_fields(self):
        config_json = self._mock_config_files()
        config = Configuration.from_json(config_json)
        combined_set = set(config.get_all_household_fields())
        correct_set = set(tuple(set(
            field.name for field in allocation.DEFAULT_HOUSEHOLD_FIELDS).union(
                set(config.household_fields))))
        self.assertEqual(combined_set, correct_set)
