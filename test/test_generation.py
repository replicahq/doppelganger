from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from mock import MagicMock

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
            mock_person('b', '35-64', 'F', '40k+')
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

    def _check_output(self, dataframe):
        self.assertSequenceEqual(
            dataframe['tract'].tolist(), ('tract1', 'tract1', 'tract2', 'tract2'))
        self.assertSequenceEqual(
            dataframe[inputs.SERIAL_NUMBER.name].tolist(), ('b', 'b', 'b', 'b'))
        self.assertSequenceEqual(dataframe['repeat_index'].tolist(), (0, 1, 0, 1))

    def test_generate_persons_simple(self):
        person_model = self._mock_model(
            [inputs.AGE.name, inputs.SEX.name],
            # Returns two people regardles of count passed in
            generated=[('35-64', 'F'), ('35-64', 'F')]
        )
        allocations = self._mock_allocated()
        population = Population.generate(
            allocations, person_model, MagicMock())

        evidence = ((inputs.AGE.name, '35-64'), (inputs.SEX.name, 'F'))

        person_model.generate.assert_called_with(
            'one_bucket', evidence, count=2)
        self._check_output(population.generated_people)
        self.assertSequenceEqual(
            population.generated_people[
                inputs.AGE.name].tolist(), ['35-64'] * 4
        )

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
        self._check_output(population.generated_households)
