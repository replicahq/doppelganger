# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import unittest
import math
import sys

import pandas
from mock import patch, mock_open
import numpy

from doppelganger import (
    bayesnets,
    inputs,
    datasource,
    BayesianNetworkModel,
    Preprocessor
)


class BayesNetTests(unittest.TestCase):

    def _one_person_house(self):
        return inputs.num_people_discrete(num_people=1)

    def _two_person_house(self):
        return inputs.num_people_discrete(num_people=2)

    def _mock_household_input(self):
        def mock_household(serialno, num_people, num_vehicles, income):
            return {
                'serial_number': serialno,
                'num_people': num_people,
                'num_vehicles': num_vehicles,
                'household_income': income
            }
        return datasource.CleanedData(pandas.DataFrame([
            mock_household('a', '1', '1', '<=0'),
            mock_household('b', '2', '6+', '40k+'),
            mock_household('c', '1', 'None', '<=0')
        ]))

    def _mock_person(self, serialno, age, sex, income, weight=1):
        return {
            'serial_number': serialno,
            'age': age,
            'sex': sex,
            'individual_income': income,
            'person_weight': weight
        }

    def _mock_people_input(self, weight=1):
        return datasource.CleanedData(pandas.DataFrame([
            self._mock_person('a', '0-17', 'M', '<=0', weight),
            self._mock_person('b', '35-64', 'F', '40k+', weight),
            self._mock_person('b', '65+', 'M', '0-40k', weight),
            self._mock_person('c', '18-34', 'M', '<=0', weight)
        ]))

    def _mock_nodes(self):
        return ('age', 'sex', 'income')

    def _mock_edges(self):
        return {
            'age': ['sex', 'income'],
            'sex': ['income']
        }

    def _household_fields(self):
        return [inputs.HOUSEHOLD_INCOME.name, inputs.NUM_VEHICLES.name]

    def _person_fields(self):
        return [
            inputs.AGE.name,
            inputs.SEX.name,
            inputs.INDIVIDUAL_INCOME.name
        ]

    def _mock_persons_missing(self):
        return datasource.CleanedData(pandas.DataFrame([
            self._mock_person('d', '0-17', 'M', inputs.UNKNOWN),
            self._mock_person('e', '35-64', 'F', '40k+'),
            self._mock_person('e', inputs.UNKNOWN, 'M', '0-40k'),
            self._mock_person('f', '18-34', 'M', '<=0')
        ]))

    def _person_structure(self):
        return ((), (0,), (0, 1))

    def _household_structure(self):
        return ((), (0,))

    def _person_segmenter(self):
        serialno_to_household = {
            'a': self._one_person_house(),
            'b': self._two_person_house(),
            'c': self._one_person_house(),
            'd': self._one_person_house(),
            'e': self._two_person_house(),
            'f': self._one_person_house()
        }

        return lambda x: serialno_to_household[x['serial_number']]

    def _household_segmenter(self):
        return lambda x: x['num_people']

    def _mock_household_collection(self):
        household_data = self._mock_household_input()
        people_data = self._mock_people_input()
        people_training_data = bayesnets.SegmentedData.from_data(
            people_data, self._person_fields(), 'person_weight', self._person_segmenter()
        )
        household_training_data = bayesnets.SegmentedData.from_data(
            household_data, self._household_fields(), segmenter=self._household_segmenter()
        )
        household_model = bayesnets.BayesianNetworkModel.train(
            household_training_data, self._household_structure(), self._household_fields()
        )
        person_model = bayesnets.BayesianNetworkModel.train(
            people_training_data, self._person_structure(), self._person_fields()
        )
        return household_model, person_model

    def test_read_households(self):
        household_data = self._mock_household_input()
        training_data = bayesnets.SegmentedData.from_data(
            household_data, self._household_fields(), segmenter=self._household_segmenter()
        )
        self.assertEqual(training_data.num_rows_data(), 3)
        expected_types = set([self._one_person_house(), self._two_person_house()])
        self.assertSetEqual(expected_types, set(training_data.types()))

    def test_read_people(self):
        people_data = self._mock_people_input()
        training_data = bayesnets.SegmentedData.from_data(
            people_data, self._person_fields(), 'person_weight', self._person_segmenter()
        )

        self.assertEqual(len(training_data.type_to_data[self._one_person_house()]), 2)
        self.assertEqual(len(training_data.type_to_data[self._two_person_house()]), 2)

    def test_read_people_weighted(self):
        people_data = self._mock_people_input(weight=3)
        training_data = bayesnets.SegmentedData.from_data(
            people_data, self._person_fields(), 'person_weight', self._person_segmenter()
        )
        self.assertEqual(training_data.num_rows_data(), 12)

    def test_generate_person(self):
        _, person_model = self._mock_household_collection()

        # Check that values are properly fixed based on
        # observation
        person = person_model.generate(self._two_person_house(), ((str('age'), str('65+')),),)[0]
        age_index = self._person_fields().index(inputs.AGE.name)
        self.assertEqual(person[age_index], '65+')

        person = person_model.generate(self._two_person_house(), ((str('sex'), str('F')),),)[0]
        sex_index = self._person_fields().index(inputs.SEX.name)
        self.assertEqual(person[sex_index], 'F')

    def test_generate_person_no_data(self):
        _, person_model = self._mock_household_collection()
        with self.assertRaises(ValueError):
            # We never observed any sex '2' in single-person
            # households, so we should not be able to generate it
            person = person_model.generate(self._one_person_house(), ((str('sex'), str('F')),),)
            sex_index = self._person_fields().index(inputs.SEX.name)
            self.assertEqual(person[sex_index], 'F')

    def test_generate_person_multiple(self):
        _, person_model = self._mock_household_collection()
        people = person_model.generate(
            self._two_person_house(), ((str('age'), str('65+')),), count=5
        )
        self.assertEqual(len(people), 5)
        age_index = self._person_fields().index(inputs.AGE.name)
        for person in people:
            self.assertEqual(person[age_index], '65+')

    def _check_household_generate(self, household_model):
        household = household_model.generate(
            self._two_person_house(), ((inputs.HOUSEHOLD_INCOME.name, str('40k+')),))[0]
        vehicle_count_index = self._household_fields().index(inputs.NUM_VEHICLES.name)
        self.assertEqual(household[vehicle_count_index], '6+')

    def test_generate_household(self):
        household_model, _ = self._mock_household_collection()

        # Check that values are properly fixed based on
        # observation
        self._check_household_generate(household_model)

    def test_load_bayes_net(self):
        nodes = self._mock_nodes()
        edges = self._mock_edges()
        structure = bayesnets.define_bayes_net_structure(nodes, edges)
        # pomegranate requires the use of tuples for edges but edges aren't ordered
        self.assertSequenceEqual(
            [set(node) for node in structure],
            [set(), set((0,)), set((0, 1))],
        )

    def test_train_bayes_net_from_generated_structure(self):
        '''Ensure the structures created by define_bayes_net_structure
        are compatible with pomegranate
        '''
        nodes = self._mock_nodes()
        edges = self._mock_edges()
        person_structure = bayesnets.define_bayes_net_structure(nodes, edges)
        person_training_data = self._mock_people_input()
        training_data = bayesnets.SegmentedData.from_data(
            person_training_data, self._person_fields(), segmenter=self._person_segmenter()
        )
        self.assertEquals(
            type(
                BayesianNetworkModel.train(
                    training_data,
                    person_structure,
                    self._person_fields,
                    prior_data=None
                )
            ),
            bayesnets.BayesianNetworkModel
        )

    def test_to_from_json(self):
        household_model, _ = self._mock_household_collection()
        household_string = household_model.to_json()
        household_model_new = BayesianNetworkModel.from_json(household_string)
        self.assertSequenceEqual(household_model.fields, household_model_new.fields)
        self._check_household_generate(household_model_new)

    def test_update_missing(self):
        _, person_model = self._mock_household_collection()
        missing_data = self._mock_persons_missing()
        training_data = bayesnets.SegmentedData.from_data(
            missing_data, self._person_fields(), segmenter=self._person_segmenter()
        )
        person_model.update(training_data)
        person = person_model.generate(self._two_person_house(), ((str('age'), str('65+')),),)[0]
        age_index = self._person_fields().index(inputs.AGE.name)
        self.assertEqual(person[age_index], '65+')

    def test_update_missing_iterations_inertia(self):
        _, person_model = self._mock_household_collection()
        missing_data = self._mock_persons_missing()
        training_data = bayesnets.SegmentedData.from_data(
            missing_data, self._person_fields(), segmenter=self._person_segmenter(),
        )
        person_model.update(training_data, max_iterations=5, inertia=.5)
        person = person_model.generate(self._two_person_house(), ((str('age'), str('65+')),),)[0]
        age_index = self._person_fields().index(inputs.AGE.name)
        self.assertEqual(person[age_index], '65+')

    def test_prior_creation(self):
        all_values = bayesnets.generate_laplace_prior_data(
            (inputs.AGE.name, inputs.SEX.name), Preprocessor())
        expected = {
            ('0-17', 'M'),
            ('18-34', 'M'),
            ('35-64', 'M'),
            ('65+', 'M'),
            ('0-17', 'F'),
            ('18-34', 'F'),
            ('35-64', 'F'),
            ('65+', 'F'),
        }
        self.assertSetEqual(expected, all_values)

    def test_generate_with_prior(self):
        network = BayesianNetworkModel.train(
            bayesnets.SegmentedData({'one_bucket': [('35-64', 'F', '40k+')]}),
            self._person_structure(),
            self._person_fields(),
            prior_data={('35-64', 'F', '40k+')}
        )
        person = network.generate('one_bucket', ())[0]
        self.assertEquals(person, ('35-64', 'F', '40k+'))

    def test_generate_with_prior_non_existing(self):
        network = BayesianNetworkModel.train(
            bayesnets.SegmentedData({'one_bucket': [('65+', 'M', '40k+')]}),
            self._person_structure(),
            self._person_fields(),
            prior_data={('35-64', 'F', '40k+')}
        )
        person = network.generate('one_bucket', ((str('sex'), str('F')),))[0]
        self.assertEquals(person, ('35-64', 'F', '40k+'))

        person = network.generate('one_bucket', ((str('sex'), str('M')),))[0]
        self.assertEquals(person, ('65+', 'M', '40k+'))

    def test_evaluate_network(self):
        people_data = self._mock_people_input()
        people_training_data = bayesnets.SegmentedData.from_data(
            people_data, self._person_fields(), 'person_weight',
            self._person_segmenter()
        )
        person_model = bayesnets.BayesianNetworkModel.train(
            people_training_data, self._person_structure(),
            self._person_fields()
        )
        likelihoods = person_model.log_likelihood(people_training_data)
        one_person = math.exp(likelihoods[self._one_person_house()])
        self.assertAlmostEqual(one_person, .25)
        two_person = math.exp(likelihoods[self._two_person_house()])
        self.assertAlmostEqual(two_person, .25)

    def test_evaluate_zero_likelihood(self):
        people_data = self._mock_people_input()
        people_training_data = bayesnets.SegmentedData.from_data(
            people_data, self._person_fields(), 'person_weight', self._person_segmenter()
        )
        person_model = bayesnets.BayesianNetworkModel.train(
            people_training_data, self._person_structure(), self._person_fields()
        )
        zero_prob_data = bayesnets.SegmentedData({self._one_person_house(): [('0-17', 'F', '<=0')]})
        likelihoods = person_model.log_likelihood(zero_prob_data)
        one_person = math.exp(likelihoods[self._one_person_house()])
        self.assertAlmostEqual(one_person, 0)

    def test_generate_dataframes(self):
        _, person_model = self._mock_household_collection()
        dataframes = person_model.probabilities_as_dataframes()
        self.assertSetEqual(set(dataframes.keys()), {'1', '2'})

        def _check_dataframe(dataframe, expected_cols,
                             expected_contents, expected_rows=None):
            self.assertSetEqual(set(dataframe), expected_cols)
            if expected_rows is not None:
                # Doesn't need to be checked for series
                self.assertSetEqual(set(dataframe.index), expected_rows)
            numpy.testing.assert_array_equal(dataframe, expected_contents)

        # Check single-person state 0
        expected_columns = {'0-17', '18-34'}
        _check_dataframe(dataframes['1'][0], expected_columns, [[.5, .5]])

        # Check single-person state 2
        expected_rows = {('0-17', 'M'), ('18-34', 'M',)}
        expected_columns = {'<=0'}
        _check_dataframe(
            dataframes['1'][2], expected_columns, [[1.], [1.]], expected_rows
        )

    def test_read_write(self):
        household_model, _ = self._mock_household_collection()

        def _check_network(household_model_new):
            self.assertSequenceEqual(household_model.fields, household_model_new.fields)
            self._check_household_generate(household_model_new)

        builtin_module_name = 'builtins' if sys.version_info.major == 3 else '__builtin__'
        with patch('{}.open'.format(builtin_module_name), new_callable=mock_open()) as open_mock:
            household_model.write('file')
            open_mock.assert_called_once_with('file', 'w')
            # Check that correct json is written
            json_net = open_mock.return_value.__enter__.return_value.write.call_args[0][0]
            household_model_new = BayesianNetworkModel.from_json(
                json_net, self._household_segmenter())
            _check_network(household_model_new)

            # Check that json is correctly read
            open_mock.return_value.__enter__.return_value.read.return_value = json_net
            household_model_new = BayesianNetworkModel.from_file(
                'file', self._household_segmenter())
            _check_network(household_model_new)
