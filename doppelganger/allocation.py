# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from collections import defaultdict, namedtuple
import numpy as np
import pandas

from doppelganger.listbalancer import (
    balance_multi_cvx, discretize_multi_weights
)
from doppelganger import inputs

# These are the minimum fields needed to allocate households
DEFAULT_PERSON_FIELDS = {
    inputs.SERIAL_NUMBER,
    inputs.AGE,
    inputs.SEX,
    inputs.PERSON_WEIGHT,
    inputs.PUMA
}


DEFAULT_HOUSEHOLD_FIELDS = {
    inputs.SERIAL_NUMBER,
    inputs.NUM_PEOPLE,
    inputs.HOUSEHOLD_WEIGHT,
    inputs.PUMA
}


CountInformation = namedtuple('CountInformation', ['tract', 'count'])


class HouseholdAllocator(object):

    @staticmethod
    def from_csvs(households_csv, persons_csv):
        """Load saved household and person allocations.

        Args:
            households_csv (unicode): path to households file
            persons_csv (unicode): path to persons file

        Returns:
            HouseholdAllocator: allocated persons & households_csv

        """
        allocated_households = pandas.read_csv(households_csv)
        allocated_persons = pandas.read_csv(persons_csv)
        return HouseholdAllocator(allocated_households, allocated_persons)

    @staticmethod
    def from_cleaned_data(marginals, households_data, persons_data):
        """Allocate households based on the given data.

        marginals (Marginals): controls to match when allocating
        households_data (CleanedData): data about households.  Must contain
            DEFAULT_HOUSEHOLD_FIELDS.
        persons_data (CleanedData): data about persons.  Must contain
            DEFAULT_PERSON_FIELDS.
        """
        for field in DEFAULT_HOUSEHOLD_FIELDS:
            assert field.name in households_data.data, \
                'Missing required field {}'.format(field.name)
        for field in DEFAULT_PERSON_FIELDS:
            assert field.name in persons_data.data, \
                'Missing required field {}'.format(field.name)

        households, persons = HouseholdAllocator._format_data(
            households_data.data, persons_data.data)
        allocated_households, allocated_persons = \
            HouseholdAllocator._allocate_households(households, persons, marginals)
        return HouseholdAllocator(allocated_households, allocated_persons)

    def __init__(self, allocated_households, allocated_persons):

        self.allocated_households = allocated_households
        self.allocated_persons = allocated_persons
        self.serialno_to_counts = defaultdict(list)
        for _, row in self.allocated_households.iterrows():
            serialno = row[inputs.SERIAL_NUMBER.name]
            tract = row['tract']
            count = int(row['count'])
            self.serialno_to_counts[serialno].append(CountInformation(tract, count))

    def get_counts(self, serialno):
        """Return the information about weights for a given serial number.

        A household is repeated for a certain number of times for each tract.
        This returns a list of (tract, repeat count).  The repeat count
        indicates the number of times this serial number should be repeated in
        this tract.

        Args:
            seriano (unicode): the household's serial number

        Returns:
            list(CountInformation): the weighted repetitions for this serialno
        """
        return self.serialno_to_counts[serialno]

    def write(self, household_file, person_file):
        """Write allocated households and persons to the given files

        Args:
            household_file (unicode): path to write households to
            person_file (unicode): path to write persons to

        """
        self.allocated_households.to_csv(household_file)
        self.allocated_persons.to_csv(person_file)

    @staticmethod
    def _allocate_households(households, persons, tract_controls):
        # Only take nonzero weights
        households = households[households[inputs.HOUSEHOLD_WEIGHT.name] > 0]

        # Initial weights from PUMS
        w = households[inputs.HOUSEHOLD_WEIGHT.name].as_matrix().T

        hh_columns = ['1', '2', '3', '4+']

        hh_table = households[hh_columns].as_matrix()

        A = tract_controls.data[hh_columns].as_matrix()
        n_tracts, n_controls = A.shape
        n_samples = len(households.index.values)

        # Control importance weights
        # < 1 means not important (thus relaxing the contraint in the solver)
        mu = np.mat([1] * n_controls)

        w_extend = np.tile(w, (n_tracts, 1))
        mu_extend = np.mat(np.tile(mu, (n_tracts, 1)))
        B = np.mat(np.dot(np.ones((1, n_tracts)), A)[0])

        # Our trade-off coefficient gamma
        # Low values (~1) mean we trust our initial weights, high values
        # (~10000) mean want to fit the marginals.
        gamma = 100.

        # Meta-balancing coefficient
        meta_gamma = 100.

        hh_weights, z, q = balance_multi_cvx(
            hh_table, A, B, w_extend, gamma * mu_extend.T, meta_gamma
        )

        # We're running discretization independently for each tract
        tract_ids = tract_controls.data['TRACTCE'].values
        total_weights = np.zeros(hh_weights.shape)
        sample_weights_int = hh_weights.astype(int)
        discretized_hh_weights = discretize_multi_weights(hh_table, hh_weights)
        total_weights = sample_weights_int + discretized_hh_weights

        # Extend households and add the weights and ids
        households_extend = pandas.concat([households] * n_tracts)
        households_extend['count'] = total_weights.flatten().T
        tracts = np.repeat(tract_ids, n_samples)
        households_extend['tract'] = tracts

        return households_extend, persons

    @staticmethod
    def _format_data(households_data, persons_data):
        hp_hhs = pandas.get_dummies(
            households_data[inputs.NUM_PEOPLE.name])
        households_data = pandas.concat([households_data, hp_hhs], axis=1)

        hp_ages = pandas.get_dummies(persons_data[inputs.AGE.name])
        persons_data = pandas.concat([persons_data, hp_ages], axis=1)

        persons_trimmed = persons_data[[inputs.SERIAL_NUMBER.name, '0-17', '18-34', '35-64', '65+']]

        # Get counts we need
        persons_trimmed = persons_trimmed.groupby(
            inputs.SERIAL_NUMBER.name).sum()
        households_trimmed = households_data[[
            inputs.SERIAL_NUMBER.name,
            inputs.NUM_PEOPLE.name,
            inputs.HOUSEHOLD_WEIGHT.name, '1', '2', '3', '4+'
        ]]

        # Merge
        households_out = pandas.merge(
            households_trimmed, persons_trimmed, how='inner',
            left_on=inputs.SERIAL_NUMBER.name, right_index=True, sort=True
        )

        persons_out = persons_data[[
            inputs.SERIAL_NUMBER.name,
            inputs.SEX.name,
            inputs.AGE.name
        ]]

        return households_out, persons_out
