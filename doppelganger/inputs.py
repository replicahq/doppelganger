# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

"""Library for reading PUMS data.

"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from builtins import str

import datetime
from enum import Enum
import math

# Constant for missing data
UNKNOWN = None


class DataCategory(Enum):
    PERSON = 1
    HOUSEHOLD = 2


class DataType(object):

    def __init__(self, name, pums_name, preprocessor, type_, possible_values):
        self.name = name
        self.pums_name = pums_name
        self.preprocessor = preprocessor
        self.type_ = type_
        self.possible_values = possible_values

    def process(self, data):
        if self.preprocessor is not None:
            return self.preprocessor(data)
        return data


def is_blank(input):
    return (
        input is UNKNOWN or
        input == '' or
        isinstance(input, float) and math.isnan(input)
    )


def generate_binning_preprocessor(bins):
    if len(bins) == 0:
        return lambda x: 'all_values'
    labels = ['<={}'.format(bins[0])]
    last_label = str(bins[0])
    for elem in bins[1:]:
        bin_label = str(elem)
        labels.append(last_label + '-' + bin_label)
        last_label = bin_label
    labels.append('{}+'.format(bins[-1]))

    def generate_bin(number):
        if is_blank(number):
            return labels[0]
        for i, bin_ in enumerate(bins):
            if int(number) <= bin_:
                return labels[i]
        return labels[-1]
    return labels, generate_bin


# Default processors


def age_discrete(age):
    if is_blank(age):
        return UNKNOWN
    age = int(age)
    if age < 18:
        return '0-17'
    if age < 35:
        return '18-34'
    if age < 65:
        return '35-64'
    return '65+'


def num_people_discrete(num_people):
    if int(num_people) < 4:
        return str(int(num_people))
    return '4+'


def gender_named(input):
    if is_blank(input):
        return UNKNOWN

    input = int(input)
    if input == 1:
        return 'M'
    elif input == 2:
        return 'F'
    else:
        return UNKNOWN


def yyyy_to_age(input):
    if is_blank(input):
        return UNKNOWN
    input = str(input)
    input_year = int(input[:4])
    cur_year = datetime.date.today().year
    return cur_year - input_year


AGE = DataType('age', 'agep', age_discrete, DataCategory.PERSON,
               {'0-17', '18-34', '35-64', '65+'})

SEX = DataType('sex', 'sex', gender_named, DataCategory.PERSON, {'M', 'F'})

_income_labels, _income_preprocessor = generate_binning_preprocessor([40000])
INDIVIDUAL_INCOME = DataType(
    'individual_income', 'wagp', _income_preprocessor, DataCategory.PERSON, _income_labels
)

HOUSEHOLD_INCOME = DataType(
    'household_income', 'fincp', _income_preprocessor, DataCategory.HOUSEHOLD, _income_labels
)

NUM_VEHICLES = DataType(
    'num_vehicles', 'veh', None, DataCategory.HOUSEHOLD, {'0', '1', '2', '3', '4', '5', '6+'}
)

NUM_PEOPLE = DataType(
    'num_people', 'np', num_people_discrete, DataCategory.HOUSEHOLD, {'0', '1', '2', '3', '4+'}
)

HOUSEHOLD_WEIGHT = DataType(
    'household_weight', 'wgtp', int, DataCategory.HOUSEHOLD, None
)

PERSON_WEIGHT = DataType(
    'person_weight', 'pwgtp', int, DataCategory.PERSON, None
)

SERIAL_NUMBER = DataType('serial_number', 'serialno', None, None, None)

PUMA = DataType('puma', 'puma', str, None, None)

STATE = DataType('state', 'st', str, None, None)

PUMS_INPUTS = [
    AGE,
    SEX,
    INDIVIDUAL_INCOME,
    HOUSEHOLD_INCOME,
    NUM_VEHICLES,
    NUM_PEOPLE,
    HOUSEHOLD_WEIGHT,
    PERSON_WEIGHT,
    SERIAL_NUMBER,
    STATE,
    PUMA
]
NAME_TO_DATATYPE = {datatype.name: datatype for datatype in PUMS_INPUTS}

# Non-PUMS data types
TRACT = DataType('tract', None, None, None, None)
COUNT = DataType('count', None, None, None, None)
REPEAT_INDEX = DataType('repeat_index', None, None, None, None)
HOUSEHOLD_ID = DataType('household_id', None, None, None, None)
