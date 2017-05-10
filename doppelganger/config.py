from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import json
from doppelganger import bayesnets


class Configuration(object):
    """A configuration to population generation.

    See the sample configuration for details on how to produce a configuration
    file.
    """

    def __init__(self,
                 household_fields,
                 person_fields,
                 household_structure,
                 person_structure,
                 preprocessing_config
                 ):
        self.person_fields = person_fields
        self.household_fields = household_fields
        self.person_structure = person_structure
        self.household_structure = household_structure
        self.preprocessing_config = preprocessing_config

    @staticmethod
    def _read_net_structure(filename):
        with open(filename) as bayes_net_file_config:
            net_json = json.loads(bayes_net_file_config.read())
            return bayesnets.define_bayes_net_structure(
                net_json['nodes'], net_json['edges']
            )

    @staticmethod
    def from_json(config_json):
        """Create a Configuration from a json blob.

        Args:
            config_json (dict): the full configuration parsed json

        Returns: Configuration to be used in population generation.
        """
        network_configs = config_json['network_config_files']
        household_structure = Configuration._read_net_structure(
            network_configs['household'])
        person_structure = Configuration._read_net_structure(
            network_configs['person'])
        household_fields = tuple(config_json['household_fields'])
        person_fields = tuple(config_json['person_fields'])
        preprocessing = config_json['preprocessing']
        return Configuration(
            household_fields,
            person_fields,
            household_structure,
            person_structure,
            preprocessing
        )

    @staticmethod
    def from_file(infile):
        """Create a Configuration from a file.

        Args:
            infile (unicode): path to a json config file

        Returns: Configuration to be used in population generation.
        """
        with open(infile) as training_config_file:
            config_json = json.loads(training_config_file.read())
            return Configuration.from_json(config_json)
