# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

"""Library for modeling generative typed bayesian networks.

Each model consists of multiple networks, one for each type of input data.
For instance, a person model might have a different network for persons of
each household type. This allows the model to learn transition probabilities
for data of each type.

`SegmentedData` helps train this typed network structure by allowing users to
specify a segmentation function to segment the training data by type.
"""

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from builtins import range, str

from collections import defaultdict, Counter
import json
import itertools
import sys

import pandas
from pomegranate import BayesianNetwork


def default_segmenter(x):
    return 'one_segment'


class SegmentedData(object):
    """Segmented data for use with the segemented BayesianNetworkModel.

    Like the model itself, training data uses a mapping of type -> data.

    """

    def __init__(self, type_to_data, segmenter=None):
        self.type_to_data = type_to_data
        self.segmenter = segmenter

    @staticmethod
    def from_data(cleaned_data, fields, weight_field=None, segmenter=None):
        """Input more data.

        Args:
            cleaned_data (CleanedData): data to train on
            segmenter: function mapping a dict of data to a type for
                segmentation
            weight_field (unicode): Name of the int field that shows how much
                this row  of data should be weighted.
        """
        segmenter = segmenter or default_segmenter
        type_to_data = defaultdict(list)
        for _, row in cleaned_data.data.iterrows():
            type_ = segmenter(row)
            weight = row[weight_field] if weight_field else 1
            cleaned_row = tuple(row[fields])
            for _ in range(weight):
                type_to_data[type_].append(cleaned_row)
        return SegmentedData(type_to_data, segmenter)

    def num_rows_data(self):
        return sum(len(data) for data in self.type_to_data.values())

    def types(self):
        return self.type_to_data.keys()


class BayesianNetworkModel(object):
    """A typed Bayesian network model.

    This bayesian network model as a fixed list of nodes passed in at creation.
    It holds a separate network for each user-defined type.
    """

    def __init__(self, type_to_network, fields, segmenter=None):
        self.type_to_network = type_to_network
        self.fields = fields
        self.distribution_cache = {}
        self.segmenter = segmenter or default_segmenter

    @staticmethod
    def from_file(filename, segmenter=None):
        with open(filename) as infile:
            json_string = infile.read()
            return BayesianNetworkModel.from_json(json_string, segmenter)

    def write(self, outfilename):
        with open(outfilename, 'w') as outfile:
            json_string = self.to_json()
            outfile.write(json_string)

    def to_json(self):
        blob = {'fieldnames': self.fields}
        blob['type_to_network'] = {
            type_: json.loads(network.to_json()) for type_, network in self.type_to_network.items()
        }
        return json.dumps(blob, indent=4, sort_keys=True)

    @staticmethod
    def _df_from_conditional(probabilities):
        """
        Helper method to extract a probability table from pomegranate's json
        format for conditional probability distributions.
        """
        state_map = defaultdict(dict)
        for row in probabilities:
            evidence = tuple(row[:-2])
            value = row[-2]
            probability = float(row[-1])
            state_map[evidence][value] = probability
        return pandas.DataFrame.from_dict(state_map).transpose()

    @staticmethod
    def _df_from_discrete(probabilities):
        """
        Helper method to extract a probability table from pomegranate's json
        format for discrete distributions.
        """
        return pandas.DataFrame(probabilities)

    def probabilities_as_dataframes(self):
        """Create dataframes for each node in each bayesian network.

        Returns:
            dict {str -> list(DataFrame)} Dictionary from segment name to
                a list of DataFrames, one for each state of the distribution
                for that segment.  For nodes with no ancestors, the dataframe
                is just columns with the probability of each value.  For nodes
                with ancestors, the dataframe frame's rows labels are evidence
                and column labels are the values, with the cell representing
                the probability of the value given the evidence.
        """
        segment_to_states = {}
        for segment, network in self.type_to_network.items():
            state_to_dataframes = []
            for state in network.states:
                # The one to access the transition probabilities directly is
                # via the json output
                distribution = json.loads(str(state))['distribution']
                if distribution['name'] == 'ConditionalProbabilityTable':
                    probabilities = BayesianNetworkModel._df_from_conditional(
                        distribution['table'])
                else:
                    probabilities = BayesianNetworkModel._df_from_discrete(
                        distribution['parameters']
                    )
                state_to_dataframes.append(probabilities)
            segment_to_states[segment] = state_to_dataframes
        return segment_to_states

    @staticmethod
    def from_json(json_string, segmenter=None):
        """Create BayesianNetworkModel from the given json blob in string format

        Args:
            json_string (unicode): the string created by `from_json`

        Returns:
            BayesianNetworkModel: generative model equivalent to stored model
        """
        json_blob = json.loads(json_string)
        type_to_network = {}
        for type_, network_json in json_blob['type_to_network'].items():
            type_to_network[type_] = BayesianNetwork.from_json(json.dumps(network_json))
        fields = list(json_blob['fieldnames'])
        return BayesianNetworkModel(type_to_network, fields, segmenter)

    @staticmethod
    def train(input_data, structure, fields, prior_data=None):
        """Creates bayesian networks from the given data with the given structure.

        The given data cannot contain any missing data. If called multiple
        times, the old model will be replaced.  To update the model with new
        data, see `update`.

        Args:
            input_data (SegmentedData): typed data to train on
            structure (iterable(iterable)): structure as returned from
                    define_bayes_net_structure
            fields (list(unicode)): field names to learn
            prior_data (list(data)): optional list of training samples to use
                    as a prior for each network.

        Return:
            BayesianNetworkModel: A predictive model training on the given data

        """
        type_to_network = {}
        for type_, data in input_data.type_to_data.items():
            if prior_data is not None:
                # Make defensive copy
                data = list(data) + list(prior_data)
            bayesian_network = BayesianNetwork.from_structure(data, structure)
            type_to_network[type_] = bayesian_network
        return BayesianNetworkModel(type_to_network, fields, segmenter=input_data.segmenter)

    def log_likelihood(self, training_data):
        """Compute the log likelihood of the given data given the model

        Compute the log likehood of the data for each type based on the
            Bayesian network for that type.  Assumes all data rows are
            independent.

        Args:
            input_data (SegmentedData): data whose likelihood to compute

        Returns:
            {type -> log likelihood}: The likelihood of the data for each
                type of data
        """
        type_to_likelihood = {}
        for type_, data in training_data.type_to_data.items():
            network = self.type_to_network[type_]
            data_counter = Counter(tuple(x) for x in data)
            log_likelihood = 0.0
            for data_row, count in data_counter.items():
                try:
                    log_likelihood += count * network.log_probability(data_row)
                except KeyError:
                    message = 'Data with zero likelihood {}'.format(data_row)
                    print(message, file=sys.stderr)
                    log_likelihood = float('-inf')
                    break
            type_to_likelihood[type_] = log_likelihood
        return type_to_likelihood

    def update(self, input_data, max_iterations=1, inertia=0.0):
        """Updates the distribution of a trained network based on new data.

        Missing values are accepted.  Missing values are filled in using MLE
        from the old distribution, and then the distribution is mutated to
        include the new data.

        Args:
            input_data (SegmentedData): typed data to train on
            max_iterations: (int): max number of iteratations of
                Expectation-Maximization to fill in missing data. Will run
                until hitting max_iterations or convergence, whichever happens
                first.
            inertia (float): The weight of the previous parameters of the
                model. The new parameters will roughly be old_param*inertia +
                new_param*(1-inertia), so an inertia of 0 means ignore the old
                parameters, whereas an inertia of 1 means ignore the new
                parameters. Default is 0.0.

        Return:
            BayesianNetworkModel: self, a predictive model training on the
                given data

        """
        def data_equals(old, new):
            if old is None or new is None:
                return False
            assert len(old) == len(new)
            for i in range(len(old)):
                # Compare as tuple because numpy arrays return an array
                # of bools instead of a bool on comparison.
                if tuple(old[i]) != tuple(new[i]):
                    return False
            return True

        # For each data-type, use EM to learn missing fields and update the
        # model
        for type_, data in input_data.type_to_data.items():
            data_previous = None
            data_new = None
            iteration = 0
            while not data_equals(data_previous, data_new):
                data_previous = data_new
                if iteration >= max_iterations:
                    break
                iteration += 1
                # Make a copy of the original data with the missing fields
                data_new = [list(row) for row in data]
                bayesian_network = self.type_to_network[type_]
                # Fill in missing fields
                data_new = bayesian_network.predict(data_new)
                # Update the model
                bayesian_network.fit(data_new, inertia=inertia)
        return self

    def generate(self, type_, evidence, count=1):
        """Sample from the network based on the given evidence

        Args:
            type_: user-defined type that will determine the network to use
            evidence ((field name, value), ...): any prior observed data. The
                    field names must be in the fields supplied on model
                    creation.
            count (int): the number of samples to generate
        Returns:
            tuple of data sampled, one element for each of the fields
                    supplied on model creation.

        """
        if (type_, evidence) in self.distribution_cache:
            distributions = self.distribution_cache[(type_, evidence)]
        else:
            try:
                evidence_translated = {
                    str(self.fields.index(field)): value
                    for field, value in evidence
                }
            except ValueError:
                raise ValueError('Evidence supplied not in model fields')
                # When pomegranate supports sampling directly from the BN we
                # will use that. See github issue
                # https://github.com/jmschrei/pomegranate/issues/231
            distributions = self.type_to_network[
                type_].predict_proba(evidence_translated)
            self.distribution_cache[(type_, evidence)] = distributions

        generated = tuple(
            tuple(distribution.sample() for distribution in distributions) for _ in range(count)
        )
        return generated


def define_bayes_net_structure(nodes, edges):
    """Create a bayes network based on the given configuration

    Args:
        nodes iterable(unicode): names of nodes
        edges dict(unicode -> unicode): edges in the form of parent -> child

    Returns: bayes net structure currently as a list where element i is a list
        of the indices of parents of i
    """
    child_to_parents = defaultdict(set)
    for parent, children in edges.items():
        for child in children:
            child_to_parents[child].add(parent)
    node_to_index = {name: i for i, name in enumerate(nodes)}
    structure = []
    for child in nodes:
        structure.append(tuple(node_to_index[parent] for parent in child_to_parents[child]))
    return tuple(tuple(s) for s in structure)


def generate_laplace_prior_data(fields, preprocessor):
    """Create training data for Laplace smoothing

    Generate data needed to apply Laplace smoothing to Bayesian network
    training. The value returned here can be applied to
    `BayesianNetworkModel.train`'s `prior_data` parameter to give each possible
    combination of field values an equal prior.

    Args:
        fields (iterable(string)): the names of all fields in the training data
        preprocessor (Preprocessor): preprocessor used for processing the
            training data, needed because this determines the data's possible
            values.

    Returns: (iterable(value list)) a list of data for use by
        `BayesianNetworkModel.train`'s `prior_data` parameter

    """
    all_values = (preprocessor.get_possible_values(field) for field in fields)
    return set(itertools.product(*all_values))
