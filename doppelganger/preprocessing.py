# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

"""Utility functions for discretizing values.
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import sys

import pandas

from doppelganger import inputs


class Preprocessor(object):

    def __init__(
        self, input_to_preprocessor=None, input_to_possible_values=None
    ):
        self.input_to_preprocessor = input_to_preprocessor or {}
        self.input_to_possible_values = input_to_possible_values or {}

    def process_dataframe(self, dataframe, fields, name_map):
        """Extract the data types from the given DataFrames and process them

        Args:
            dataframe (pandas.DataFrame): pums-style data
            fields (iterable): the standard names of fields to extract
            name_map (dict): map of standard name to name within this datasource

        Returns: pandas.DataFrame of the given fields, processed

        """
        cleaned_dataframe = pandas.DataFrame()
        for field_name in fields:
            if field_name in self.input_to_preprocessor:
                procesessor = self.input_to_preprocessor[field_name]
            elif field_name in inputs.NAME_TO_DATATYPE:
                procesessor = inputs.NAME_TO_DATATYPE[field_name].process
            else:
                print('Preprocessor: Unknown data field {}'.format(
                    field_name), file=sys.stderr)
                sys.exit()

            data_type = inputs.NAME_TO_DATATYPE[field_name]
            dirty_name = name_map[field_name]
            if dirty_name in dataframe:
                cleaned_dataframe[data_type.name] = dataframe[
                    dirty_name].apply(procesessor)
            elif dirty_name.upper() in dataframe:
                cleaned_dataframe[data_type.name] = dataframe[
                    dirty_name.upper()].apply(procesessor)
            else:
                print('Missing data field {}'.format(
                    field_name), file=sys.stderr)
        return cleaned_dataframe

    def get_possible_values(self, field):
        field = inputs.NAME_TO_DATATYPE[field]
        if field.name in self.input_to_possible_values:
            return self.input_to_possible_values[field.name]
        return field.possible_values

    @staticmethod
    def from_config(config):
        """Load a preprocessor from a config.

        If unspecified, assumes that data fields use PUMS names.
        """
        field_to_preprocessor = {}
        field_to_possible_values = {}
        for field in config:
            if 'bins' in config[field]:
                labels, preprocessor = inputs.generate_binning_preprocessor(
                    config[field]['bins']
                )
                field_to_preprocessor[field] = preprocessor
                field_to_possible_values[field] = labels
        return Preprocessor(field_to_preprocessor, field_to_possible_values)
