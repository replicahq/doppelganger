from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import unittest
import numpy as np

from doppelganger import (
    Configuration,
    allocation,
)


class TestConfiguration(unittest.TestCase):

    def test_get_combined_config_and_default_fields(self):

        configuration_household_fields = ('field1', 'field2', 'field3')
        combined_set = set(Configuration.get_combined_config_and_default_fields(
            allocation.DEFAULT_HOUSEHOLD_FIELDS, configuration_household_fields))

        correct_set = set(tuple(set(
            field.name for field in allocation.DEFAULT_HOUSEHOLD_FIELDS).union(
                set(configuration_household_fields))))
        np.testing.assert_equal(combined_set, correct_set)
