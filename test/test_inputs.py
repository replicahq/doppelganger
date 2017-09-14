# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import unittest

from doppelganger import inputs


class InputsTest(unittest.TestCase):

    def test_yyyy_to_age(self):
        for sample_birthday in [19490301, '19490301', '194903', 194903]:
            age = inputs.yyyy_to_age(sample_birthday)
            self.assertEqual(age, 68)

    def test_yyyy_to_age_none(self):
        for sample_birthday in ['', float('nan')]:
            age = inputs.yyyy_to_age(sample_birthday)
            self.assertEqual(age, inputs.UNKNOWN)

    def test_age_discrete(self):
        self.assertEquals(inputs.age_discrete(''), None)
        self.assertEquals(inputs.age_discrete('12'), '0-17')
        self.assertEquals(inputs.age_discrete('20'), '18-34')
        self.assertEquals(inputs.age_discrete('35'), '35-64')
        self.assertEquals(inputs.age_discrete('65'), '65+')

    def test_num_people_discrete(self):
        self.assertEquals(inputs.num_people_discrete('5'), '4+')

    def test_work_status(self):
        self.assertEquals(inputs.work_status(''), 'under16')
        self.assertEquals(inputs.work_status('1'), 'working')
        self.assertEquals(inputs.work_status('2'), 'working')
        self.assertEquals(inputs.work_status('4'), 'working')
        self.assertEquals(inputs.work_status('5'), 'working')
        self.assertEquals(inputs.work_status('3'), 'not_working')
        self.assertEquals(inputs.work_status('6'), 'not_working')

    def test_educational_attainment(self):
        self.assertEquals(inputs.educational_attainment(''), 'under3')
        self.assertEquals(inputs.educational_attainment('bb'), 'under3')

    def test_gender_named(self):
        self.assertEquals(inputs.gender_named(''), None)
        self.assertEquals(inputs.gender_named('3'), None)
        self.assertEquals(inputs.gender_named('1'), 'M')
        self.assertEquals(inputs.gender_named('2'), 'F')
