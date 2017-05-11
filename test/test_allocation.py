from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from mock import MagicMock, patch
import unittest
import pandas

from doppelganger import HouseholdAllocator


class TestAllocation(unittest.TestCase):
    def test_read_from_file(self):
        read_csv = MagicMock(return_value=pandas.DataFrame())
        with patch('pandas.read_csv', read_csv):
            allocator = HouseholdAllocator.from_csvs('households_file', 'persons_file')
        assert type(allocator) == HouseholdAllocator
        read_csv.assert_any_call('households_file')
        read_csv.assert_any_call('persons_file')

    def test_write_to_file(self):
        persons = MagicMock()
        households = MagicMock()
        allocator = HouseholdAllocator(households, persons)

        allocator.write(person_file='persons_file', household_file='households_file')
        persons.to_csv.assert_called_once_with('persons_file')
        households.to_csv.assert_called_once_with('households_file')
