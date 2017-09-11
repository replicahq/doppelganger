from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import unittest
import os
import mock
from doppelganger.scripts import download_allocate_generate
from doppelganger import config, CleanedData, inputs
import pandas


class TestDownloadAllocateGenerate(unittest.TestCase):

    def _mock_config(self):
        conf = {
            "person_fields": [
                "age",
                "sex",
                "individual_income"
            ],
            "household_fields": [
                "num_people",
                "household_income",
                "num_vehicles"
            ],
            "preprocessing": {
                "individual_income": {
                    "bins": [
                        0,
                        20000,
                        40000,
                        80000,
                        100000
                    ]
                }
            },
            "network_config_files": {
                "person": os.path.dirname(__file__)
                + "/../examples/sample_data/sample_person_bn.json",
                "household": os.path.dirname(__file__)
                + "/../examples/sample_data/sample_household_bn.json"
            },
            "version": "0"
        }
        return config.Configuration.from_json(conf)

    _mock_params = {
                    'state_id': '01',
                    'puma_id': '00001',
                    'output_dir': '.',
                    'db_host': 'host1',
                    'db_database': 'database1',
                    'db_schema': 'schema1',
                    'db_table': 'table1',
                    'db_user': 'user1',
                    'db_password': 'password1',
                    'census_api_key': 'census_key1',
                    'puma_tract_mappings': 'puma_tract_mappings',
                }

    @mock.patch('pandas.DataFrame.to_csv')
    @mock.patch('doppelganger.scripts.download_allocate_generate.CleanedData')
    @mock.patch('doppelganger.scripts.download_allocate_generate.os.path.exists')
    @mock.patch('doppelganger.scripts.download_allocate_generate.fetch_pums_data')
    def test_download_and_load_pums_data_download(self, mock_fetch_pums_data, mock_exists,
                                                  mock_CleanedData, mock_pandas_to_csv):
        '''Verify fetch_pums_data is called with the proper arguments if local pums files aren't
        found
        '''
        mock_fetch_pums_data.return_value = (CleanedData(pandas.DataFrame()),
                                             CleanedData(pandas.DataFrame()))
        configuration = self._mock_config()
        mock_exists.return_value = False

        download_allocate_generate.download_and_load_pums_data(
            output_dir=self._mock_params['output_dir'],
            state_id=self._mock_params['state_id'],
            puma_id=self._mock_params['puma_id'],
            configuration=configuration,
            db_host=self._mock_params['db_host'],
            db_database=self._mock_params['db_database'],
            db_schema=self._mock_params['db_schema'],
            db_user=self._mock_params['db_user'],
            db_password=self._mock_params['db_password']
        )
        mock_fetch_pums_data.assert_called()
        mock_fetch_pums_data.assert_called_with(
                        state_id='01',
                        puma_id='00001',
                        configuration=configuration,
                        db_host='host1',
                        db_database='database1',
                        db_schema='schema1',
                        db_user='user1',
                        db_password='password1'
                    )

    @mock.patch('pandas.DataFrame.to_csv')
    @mock.patch('doppelganger.scripts.download_allocate_generate.CleanedData')
    @mock.patch('doppelganger.scripts.download_allocate_generate.os.path.exists')
    @mock.patch('doppelganger.scripts.download_allocate_generate.fetch_pums_data')
    def test_download_and_load_pums_data_dont_download(self, mock_fetch_pums_data, mock_exists,
                                                       mock_CleanedData, mock_pandas_to_csv):
        '''Verify:
        1. database is not called when files exist already
        2. files are written out properly
        '''
        configuration = self._mock_config()
        mock_exists.return_value = True

        download_allocate_generate.download_and_load_pums_data(
            output_dir=self._mock_params['output_dir'],
            state_id=self._mock_params['state_id'],
            puma_id=self._mock_params['puma_id'],
            configuration=configuration,
            db_host=self._mock_params['db_host'],
            db_database=self._mock_params['db_database'],
            db_schema=self._mock_params['db_schema'],
            db_user=self._mock_params['db_user'],
            db_password=self._mock_params['db_password']
        )

        # Check files are being written out properly
        households_filepath = os.path.sep.join([self._mock_params['output_dir'],
                                               download_allocate_generate.FILE_PATTERN.format(
                        self._mock_params['state_id'],
                        self._mock_params['puma_id'],
                        'households_pums.csv'
                    )
                ])
        persons_filepath = os.path.sep.join([self._mock_params['output_dir'],
                                            download_allocate_generate.FILE_PATTERN.format(
                        self._mock_params['state_id'],
                        self._mock_params['puma_id'],
                        'persons_pums.csv'
                    )
                ])
        self.assertFalse(mock_fetch_pums_data.called)
        self.assertTrue(mock_CleanedData.from_csv.called)
        mock_CleanedData.from_csv.assert_any_call(households_filepath)
        mock_CleanedData.from_csv.assert_any_call(persons_filepath)

    @mock.patch('doppelganger.scripts.download_allocate_generate.SegmentedData')
    @mock.patch('doppelganger.scripts.download_allocate_generate.BayesianNetworkModel')
    def test_create_bayes_net_segmenter(self, mock_BayesianNetworkModel, mock_SegmentedData):
        configuration = self._mock_config()
        mock_persons_data = mock.Mock()
        mock_households_data = mock.Mock()
        download_allocate_generate.create_bayes_net(
                output_dir=self._mock_params['output_dir'],
                state_id=self._mock_params['state_id'],
                puma_id=self._mock_params['puma_id'],
                configuration=configuration,
                persons_data=mock_persons_data,
                households_data=mock_households_data,
                person_segmenter=download_allocate_generate.person_segmenter,
                household_segmenter=download_allocate_generate.household_segmenter
            )
        mock_SegmentedData.from_data.assert_called()

        mock_SegmentedData.from_data.assert_any_call(
                cleaned_data=mock_persons_data,
                fields=list(configuration.person_fields),
                weight_field=inputs.PERSON_WEIGHT.name,
                segmenter=download_allocate_generate.person_segmenter
                )
        mock_SegmentedData.from_data.assert_any_call(
                cleaned_data=mock_households_data,
                fields=list(configuration.household_fields),
                weight_field=inputs.HOUSEHOLD_WEIGHT.name,
                segmenter=download_allocate_generate.household_segmenter
                )

    @mock.patch('doppelganger.scripts.download_allocate_generate.SegmentedData')
    @mock.patch('doppelganger.scripts.download_allocate_generate.BayesianNetworkModel')
    def test_create_bayes_net_model(self, mock_BayesianNetworkModel, mock_SegmentedData):
        configuration = self._mock_config()
        mock_persons_data = mock.Mock()
        mock_households_data = mock.Mock()
        mock_training_data = mock.Mock()
        mock_SegmentedData.from_data.return_value = mock_training_data

        download_allocate_generate.create_bayes_net(
                output_dir=self._mock_params['output_dir'],
                state_id=self._mock_params['state_id'],
                puma_id=self._mock_params['puma_id'],
                configuration=configuration,
                persons_data=mock_persons_data,
                households_data=mock_households_data,
                person_segmenter=download_allocate_generate.person_segmenter,
                household_segmenter=download_allocate_generate.household_segmenter
            )

        mock_BayesianNetworkModel.train.assert_any_call(
                    input_data=mock_training_data,
                    structure=configuration.person_structure,
                    fields=configuration.person_fields
                )

    @mock.patch('doppelganger.scripts.download_allocate_generate.HouseholdAllocator')
    @mock.patch('doppelganger.scripts.download_allocate_generate.Marginals')
    def test_download_tracts_data_dont_download(self, mock_Marginals, mock_HouseholdAllocator):
        mock_Marginals.from_csv.return_value = True
        mock_households_data = mock.Mock()
        mock_persons_data = mock.Mock()

        download_allocate_generate.download_tract_data(
            output_dir=self._mock_params['output_dir'],
            state_id=self._mock_params['state_id'],
            puma_id=self._mock_params['puma_id'],
            census_api_key=self._mock_params['census_api_key'],
            puma_tract_mappings=self._mock_params['puma_tract_mappings'],
            households_data=mock_households_data,
            persons_data=mock_persons_data,
        )
        self.assertFalse(mock_Marginals.from_census_data.called)

    @mock.patch('builtins.open')  # python 3 compatible
    @mock.patch('doppelganger.scripts.download_allocate_generate.HouseholdAllocator')
    @mock.patch('doppelganger.scripts.download_allocate_generate.Marginals')
    def test_download_tracts_data_download(self, mock_Marginals, mock_HouseholdAllocator,
                                           mock_open):
        mock_households_data = mock.Mock()
        mock_persons_data = mock.Mock()
        mock_returned_marginals = mock.Mock()
        mock_returned_marginals.data = [1, 2, 3]  # To pass len<=1 check
        mock_Marginals.from_census_data.return_value = mock_returned_marginals
        mock_Marginals.from_csv.side_effect = [  # File only found after downloading
                    Exception('File not found'),
                    mock_returned_marginals
                ]

        download_allocate_generate.download_tract_data(
            output_dir=self._mock_params['output_dir'],
            state_id=self._mock_params['state_id'],
            puma_id=self._mock_params['puma_id'],
            census_api_key=self._mock_params['census_api_key'],
            puma_tract_mappings=self._mock_params['puma_tract_mappings'],
            households_data=mock_households_data,
            persons_data=mock_persons_data,
        )
        self.assertTrue(mock_Marginals.from_census_data.called)
        mock_HouseholdAllocator.from_cleaned_data.assert_any_call(
                    marginals=mock_returned_marginals,
                    households_data=mock_households_data,
                    persons_data=mock_persons_data
                )

    @mock.patch('doppelganger.scripts.download_allocate_generate.Population')
    def test_generate_synths(self, mock_Population):
        mock_person_model = mock.Mock()
        mock_household_model = mock.Mock()
        mock_allocator = mock.Mock()
        download_allocate_generate.generate_synthetic_people_and_households(
                output_dir=self._mock_params['output_dir'],
                state_id=self._mock_params['state_id'],
                puma_id=self._mock_params['puma_id'],
                allocator=mock_allocator,
                person_model=mock_person_model,
                household_model=mock_household_model
            )
        mock_Population.generate.assert_any_call(
                    household_allocator=mock_allocator,
                    person_model=mock_person_model,
                    household_model=mock_household_model
                )
