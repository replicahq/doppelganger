# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import unittest

from doppelganger import Preprocessor


class PreprocessingTest(unittest.TestCase):

    def test_binning_generator(self):
        bins = [0, 20000, 40000, 60000]
        config = {
            'test_input': {
                'bins': bins
            }
        }
        preprocessor = Preprocessor.from_config(config)
        preprocess = preprocessor.input_to_preprocessor['test_input']
        self.assertEqual(preprocess(0), '<=0')
        self.assertEqual(preprocess(10000), '0-20000')
        self.assertEqual(preprocess(100000), '60000+')
