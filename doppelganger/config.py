# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import json

from doppelganger import (bayesnets, allocation)

CURRENT_VERSION = '0'


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
                 preprocessing_config,
                 version
                 ):
        self.person_fields = person_fields
        self.household_fields = household_fields
        self.person_structure = person_structure
        self.household_structure = household_structure
        self.preprocessing_config = preprocessing_config
        self.version = version

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

        assert 'version' in config_json, "The config file is missing a 'version' field"
        assert config_json['version'] == CURRENT_VERSION, \
            "The config file version is incorrect. \
            Please upgrade your config file to version {}.".format(CURRENT_VERSION)
        network_configs = config_json['network_config_files']
        household_structure = Configuration._read_net_structure(
            network_configs['household'])
        person_structure = Configuration._read_net_structure(
            network_configs['person'])
        household_fields = tuple(config_json['household_fields'])
        person_fields = tuple(config_json['person_fields'])
        preprocessing = config_json['preprocessing']
        version = config_json['version']
        return Configuration(
            household_fields,
            person_fields,
            household_structure,
            person_structure,
            preprocessing,
            version
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

    def get_all_person_fields(self):
        """Create a tuple of combined config and the default person fields.

        Returns: Combination of allocation default person
            fields and configuration person fields as tuple
        """

        default_set = set(field.name for field in allocation.DEFAULT_PERSON_FIELDS)
        config_set = set(self.person_fields)
        return tuple(default_set.union(config_set))

    def get_all_household_fields(self):
        """Create a tuple of combined config and the default household fields.

        Returns: Combination of allocation default household
            fields and configuration household fields as tuple
        """

        default_set = set(field.name for field in allocation.DEFAULT_HOUSEHOLD_FIELDS)
        config_set = set(self.household_fields)
        return tuple(default_set.union(config_set))
