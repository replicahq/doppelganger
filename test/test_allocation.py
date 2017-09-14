# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from mock import MagicMock, patch
import unittest
import pandas

from doppelganger import (HouseholdAllocator, CleanedData, Marginals)


class TestAllocation(unittest.TestCase):
    def _mock_person_data(self):
        # Just the top 10 lines of a sample PUMS file, counts will NOT line up with marginals.
        return pandas.DataFrame([
            ['0', '00901', '0-20000', '18-34', 'F', '29', '1010395', 20],
            ['1', '00901', '20000-40000', '65+', 'F', '29', '1014317', 131],
            ['2', '00901', '40000-80000', '35-64', 'M', '29', '1019591', 88],
            ['3', '00901', '20000-40000', '35-64', 'F', '29', '1019591', 56],
            ['4', '00901', '<=0', '18-34', 'F', '29', '1028480', '16'],
            ['5', '00901', '20000-40000', '18-34', 'M', '29', '1029939', 65],
            ['6', '00901', '<=0', '0-17', 'M', '29', '1029939', 92],
            ['7', '00901', '0-20000', '18-34', 'F', '29', '1029939', 137],
            ['8', '00901', '<=0', '35-64', 'F', '29', '103719', 93],
            ['9', '00901', '20000-40000', '35-64', 'M', '29', '1038824', 165],
            ['10', '00901', '<=0', '65+', 'F', '29', '1045157', 38]
        ], columns=['row', 'puma', 'individual_income', 'age', 'sex', 'state',
                    'serial_number', 'person_weight'])

    def _mock_household_data(self):
        return pandas.DataFrame([
            ['0', '00901', '1', '29', '<=40000', '0', '1010395', 0],
            ['1', '00901', '1', '29', '<=40000', '1', '1014317', 130],
            ['2', '00901', '2', '29', '40000+', '3+', '1019591', 88],
            ['3', '00901', '1', '29', '<=40000', '0', '1028480', 0],
            ['4', '00901', '3', '29', '<=40000', '2', '1029939', 65],
            ['5', '00901', '1', '29', '<=40000', '1', '103719', 93],
            ['6', '00901', '1', '29', '<=40000', '3+', '1038824', 165],
            ['7', '00901', '2', '29', '40000+', '2', '1045157', 39],
            ['8', '00901', '3', '29', '40000+', '2', '1047723', 23],
            ['9', '00901', '1', '29', '<=40000', '3+', '1048234', 73],
            ['10', '00901', '2', '29', '40000+', '3+', '1048646', 143]
        ], columns=['row', 'puma', 'num_people', 'state', 'household_income',
                    'num_vehicles', 'serial_number', 'household_weight'])

    def _mock_tract_data(self):
        return pandas.DataFrame([
            ['0', '29', '047', '00901', '020801',
                2768, 792, 372, 1068, 536, 773, 40, 1676, 831, 1471, 1487, 1115, 2825],
            ['1', '29', '047', '00901', '021307',
                2283, 720, 469, 506, 588, 640, 136, 1335, 939, 1762, 1309, 415, 2416],
            ['2', '29', '047', '00901', '021309',
                1430, 211, 193, 453, 573, 213, 8, 973, 918, 1752, 605, 297, 1953],
            ['3', '29', '047', '00901', '021310',
                2483, 297, 418, 777, 991, 227, 0, 2618, 1308, 2821, 1440, 318, 3281],
            ['4', '29', '047', '00901', '021401',
                1449, 343, 262, 508, 336, 340, 35, 842, 859, 956, 1462, 633, 1908],
            ['5', '29', '047', '00901', '021403',
                1289, 506, 207, 426, 150, 375, 44, 692, 264, 587, 650, 412, 1143],
            ['6', '29', '047', '00901', '021404',
                1803, 463, 241, 709, 390, 304, 10, 942, 850, 1036, 936, 937, 1861],
            ['7', '29', '047', '00901', '021600',
                2754, 461, 314, 1024, 955, 410, 13, 1637, 1812, 2482, 1437, 760, 3727],
            ['8', '29', '047', '00901', '021701',
                2053, 397, 372, 799, 485, 381, 45, 1427, 872, 1533, 1211, 998, 2173],
            ['9', '29', '047', '00901', '021702',
                2368, 758, 191, 838, 581, 764, 15, 986, 968, 1573, 1948, 883, 2089],
            ['10', '29', '047', '00901', '021803',
                2549, 287, 446, 786, 1030, 625, 0, 2772, 1054, 2753, 1534, 420, 3243],
            ['11', '29', '047', '00901', '021804',
                2325, 262, 300, 1073, 690, 208, 0, 1297, 1882, 1633, 791, 818, 3284],
            ['12', '29', '047', '00901', '021805',
                2863, 413, 485, 1040, 925, 321, 0, 2270, 1723, 2698, 1450, 934, 3539],
            ['13', '29', '047', '00901', '021806',
                1520, 239, 311, 700, 270, 224, 5, 571, 983, 949, 582, 744, 1803],
            ['14', '29', '047', '00901', '021900',
                2239, 340, 456, 759, 684, 221, 22, 1706, 1461, 1692, 1235, 536, 2949],
            ['15', '29', '047', '00901', '022000',
                1959, 621, 281, 580, 477, 472, 0, 871, 1076, 1396, 732, 624, 2234],
            ['16', '29', '047', '00901', '022200',
                1349, 240, 205, 570, 334, 235, 18, 870, 737, 1042, 1008, 474, 1309],
            ['17', '29', '047', '00901', '022301',
                1195, 343, 169, 465, 218, 325, 52, 601, 489, 621, 648, 378, 1198],
            ['18', '29', '047', '00901', '022302',
                2102, 439, 429, 672, 562, 204, 14, 1691, 1093, 1373, 1025, 872, 2566]
        ], columns=[
            'row', 'STATEFP', 'COUNTYFP', 'PUMA5CE', 'TRACTCE', 'num_people_count',
            'num_people_1', 'num_people_3', 'num_people_2', 'num_people_4+', 'num_vehicles_1',
            'num_vehicles_0', 'num_vehicles_2', 'num_vehicles_3+', 'age_0-17', 'age_18-34',
            'age_65+', 'age_35-64'
        ])

    def test_from_cleaned_data(self):
        # Prepare pums data
        households_data = CleanedData(self._mock_household_data())
        persons_data = CleanedData(self._mock_person_data())
        # Prepare marginals
        marginals = Marginals(self._mock_tract_data())
        allocator = HouseholdAllocator.from_cleaned_data(marginals, households_data, persons_data)
        self.assertTrue(allocator)
        expected_shape = (114, 17)
        self.assertEquals(allocator.allocated_households.shape, expected_shape)
        expected_columns = [
                u'serial_number', u'num_people', u'num_vehicles', u'household_weight',
                u'num_people_1', u'num_people_2', u'num_people_3', u'num_vehicles_0',
                u'num_vehicles_1', u'num_vehicles_2', u'num_vehicles_3+', u'age_0-17',
                u'age_18-34', u'age_65+', u'age_35-64', u'count', u'tract'
            ]
        self.assertEquals(set(allocator.allocated_households.columns.tolist()),
                          set(expected_columns))

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
