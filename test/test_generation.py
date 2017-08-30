# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from mock import MagicMock, patch

import unittest
import pandas

from doppelganger import inputs, Population, HouseholdAllocator


class TestPopulationGen(unittest.TestCase):

    def _mock_allocated(self):
        def mock_person(serialno, age, sex, income):
            return {
                'serial_number': serialno,
                'age': age,
                'sex': sex,
                'individual_income': income
            }
        allocated_persons = pandas.DataFrame([
            mock_person('b', '35-64', 'F', '40k+'),
            mock_person('b', '35-64', 'M', 'None'),
        ])

        def mock_household(serialno, num_people, num_vehicles, income, count, tract):
            return {
                'serial_number': serialno,
                'num_people': num_people,
                'num_vehicles': num_vehicles,
                'household_income': income,
                'count': count,
                'tract': tract,
            }
        allocated_households = pandas.DataFrame([
            mock_household('b', '6+', '2', '40k+', count=2, tract='tract1'),
            mock_household('b', '6+', '2', '40k+', count=2, tract='tract2'),
        ])
        return HouseholdAllocator(allocated_households, allocated_persons)

    def _mock_model(self, fields, generated):
        model = MagicMock()
        model.fields = fields
        model.segmenter = MagicMock(return_value='one_bucket')
        model.generate = MagicMock(
            return_value=generated)
        return model

    def _check_household_output(self, dataframe):
        self.assertSequenceEqual(
            dataframe[inputs.TRACT.name].tolist(), ('tract1', 'tract1', 'tract2', 'tract2'))
        self.assertSequenceEqual(
            dataframe[inputs.SERIAL_NUMBER.name].tolist(), ('b', 'b', 'b', 'b'))
        self.assertSequenceEqual(dataframe[inputs.REPEAT_INDEX.name].tolist(), (0, 1, 0, 1))
        self.assertEqual(dataframe[inputs.HOUSEHOLD_ID.name].tolist()[0], 'tract1-b-0')

    def _check_person_output(self, dataframe):
        self.assertSequenceEqual(
            dataframe[inputs.TRACT.name].tolist(), ('tract1', 'tract1', 'tract2', 'tract2',
                                                    'tract1', 'tract1', 'tract2', 'tract2'))
        self.assertSequenceEqual(
            dataframe[inputs.REPEAT_INDEX.name].tolist(), (0, 1, 0, 1, 0, 1, 0, 1))
        self.assertSequenceEqual(dataframe[inputs.AGE.name].tolist(), ['35-64'] * 8)
        self.assertEqual(dataframe[inputs.HOUSEHOLD_ID.name].tolist()[0], 'tract1-b-0')

    def test_generate_persons_simple(self):
        person_model = self._mock_model(
            [inputs.AGE.name, inputs.SEX.name],
            # Returns two people regardles of count passed in
            generated=[('35-64', 'F'), ('35-64', 'F')]
        )
        allocations = self._mock_allocated()
        population = Population.generate(
            allocations, person_model, MagicMock())

        evidence = ((inputs.AGE.name, '35-64'), (inputs.SEX.name, 'M'))

        person_model.generate.assert_called_with(
            'one_bucket', evidence, count=2)
        self._check_person_output(population.generated_people)

    def test_generate_households_simple(self):
        household_model = self._mock_model(
            [inputs.NUM_PEOPLE.name],
            generated=[('6+',), ('6+',)]
        )
        allocations = self._mock_allocated()
        population = Population.generate(
            allocations, MagicMock(), household_model)

        evidence = ((inputs.NUM_PEOPLE.name, '6+'),)

        household_model.generate.assert_called_with(
            'one_bucket', evidence, count=2)

        self.assertIn(inputs.NUM_PEOPLE.name, population.generated_households)
        self._check_household_output(population.generated_households)

    def test_read_from_file(self):
        read_csv = MagicMock(return_value=pandas.DataFrame())
        with patch('pandas.read_csv', read_csv):
            population = Population.from_csvs('persons_file', 'households_file')
        assert type(population) == Population
        read_csv.assert_any_call('households_file')
        read_csv.assert_any_call('persons_file')

    def test_write_to_file(self):
        persons = MagicMock()
        households = MagicMock()
        population = Population(persons, households)

        population.write(persons_outfile='persons_file', households_outfile='households_file')
        persons.to_csv.assert_called_once_with('persons_file')
        households.to_csv.assert_called_once_with('households_file')
