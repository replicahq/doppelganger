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


class WorkStatus(object):
    UNDER_16 = 'under16'
    NOT_WORKING = 'not_working'
    WORKING = 'working'


class EducationalAttainment(object):
    UNDER_3 = 'under3'
    NO_SCHOOL = 'no_school'
    K12 = 'k-12'
    HIGH_SCHOOL = 'high_school'
    SOME_COLLEGE = 'some_college'
    BACHELORS_DEGREE = 'bachelors_degree'
    ADVANCED_DEGREE = 'advanced_degree'


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
    if is_blank(num_people):
        return '0'  # ACS has no equivalent to the PUMS '' value
    if int(num_people) < 4:
        return str(int(num_people))
    return '4+'


def work_status(code):
    ''' Employment status recode (PUMS 2015 code: ESR)
        b .N/A (less than 16 years old)
        1 .Civilian employed, at work
        2 .Civilian employed, with a job but not at work
        3 .Unemployed
        4 .Armed forces, at work
        5 .Armed forces, with a job but not at work
        6 .Not in labor force
    '''
    if is_blank(code):
        return WorkStatus.UNDER_16
    if code == '1' or code == '2' or code == '4' or code == '5':
        return WorkStatus.WORKING
    if code == '3' or code == '6':
        return WorkStatus.NOT_WORKING


def educational_attainment(code):
    ''' Educational attainment (PUMS 2015 code: SCHL)
    Pums_Value Category
        bb     N/A (less than 3 years old)
        01     No schooling completed
        02     Nursery school, preschool
        03     Kindergarten
        04     Grade 1
        05     Grade 2
        06     Grade 3
        07     Grade 4
        08     Grade 5
        09     Grade 6
        10     Grade 7
        11     Grade 8
        12     Grade 9
        13     Grade 10
        14     Grade 11
        15     12th grade - no diploma
        16     Regular high school diploma
        17     GED or alternative credential
        18     Some college, but less than 1 year
        19     1 or more years of college credit, no degree
        20     Associate's degree
        21     Bachelor's degree
        22     Master's degree
        23     Professional degree beyond a bachelor's degree
        24     Doctorate degree
    '''
    if is_blank(code) or code == 'bb':
        return EducationalAttainment.UNDER_3
    if code == '01':
        return EducationalAttainment.NO_SCHOOL
    if (  # Including preschool with k-12 to avoid extra category in model
            code == '02' or code == '03' or code == '04' or code == '05' or code == '06' or
            code == '07' or code == '08' or code == '09' or code == '10' or code == '11' or
            code == '12' or code == '13' or code == '14' or code == '15'
       ):
        return EducationalAttainment.K12
    if code == '16' or code == '17':
        return EducationalAttainment.HIGH_SCHOOL
    if code == '18' or code == '19' or code == '20':
        return EducationalAttainment.SOME_COLLEGE
    if code == '21':
        return EducationalAttainment.BACHELORS_DEGREE
    if code == '22' or code == '23' or code == '24':
        return EducationalAttainment.ADVANCED_DEGREE
    return UNKNOWN


def num_vehicles_discrete(num_vehicles):
    if is_blank(num_vehicles):
        return '0'  # ACS has no equivalent to the PUMS '' value
    if int(num_vehicles) < 3:
        return str(int(num_vehicles))
    return '3+'


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

WORKING = DataType('working', 'esr', work_status, DataCategory.PERSON, {'under-16', 'Y', 'N'})

EDUCATION = DataType('education', 'schl', educational_attainment, DataCategory.PERSON,
                     {'under-3', 'none', 'k-12', 'high-school', 'some-college', 'bachelors-degree',
                      'advanced-degree'})

_income_labels, _income_preprocessor = generate_binning_preprocessor([40000])
INDIVIDUAL_INCOME = DataType(
    'individual_income', 'wagp', _income_preprocessor, DataCategory.PERSON, _income_labels
)

HOUSEHOLD_INCOME = DataType(
    'household_income', 'fincp', _income_preprocessor, DataCategory.HOUSEHOLD, _income_labels
)

NUM_VEHICLES = DataType(
    'num_vehicles', 'veh', num_vehicles_discrete, DataCategory.HOUSEHOLD,
    {'0', '1', '2', '3+'}
)

NUM_PEOPLE = DataType(
    'num_people', 'np', num_people_discrete, DataCategory.HOUSEHOLD,
    {'1', '2', '3', '4+'}
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

TRAFFIC_ANALYSIS_ZONE = DataType('taz', None, str, None, None)

PUMS_INPUTS = [
    AGE,
    SEX,
    WORKING,
    EDUCATION,
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
