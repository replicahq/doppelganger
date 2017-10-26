# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import pandas as pd
import numpy as np
from collections import OrderedDict
import os
from enum import Enum

from doppelganger import marginals

FILE_PATTERN = 'state_{}_puma_{}_{}'


class ErrorStat(Enum):
    ROOT_MEAN_SQUARED_ERROR = 1
    ROOT_SQUARED_ERROR = 2
    ABSOLUTE_PCT_ERROR = 3


class AccuracyException(Exception):
    pass


class Accuracy(object):
    def __init__(self, person_pums, household_pums, marginal_data,
                 generated_persons, generated_households, marginal_variables, use_all_marginals):
        self.comparison_dataframe = Accuracy._comparison_dataframe(
            person_pums,
            household_pums,
            marginal_data,
            generated_persons,
            generated_households,
            marginal_variables,
            use_all_marginals
        )

    @staticmethod
    def from_doppelganger(
                cleaned_data_persons,
                cleaned_data_households,
                marginal_data,
                population,
                marginal_variables=[],
                use_all_marginals=True
            ):
        '''Initialize an accuracy object from doppelganger objects
            cleaned_data_persons (doppelgange.DataSource.CleanedData) - pums person data
            cleaned_data_households (doppelgange.DataSource.CleanedData) - pums household data
            marginal_data (doppelganger.Marginals) - marginal data (usually census)
            population (doppelganger.Population) - Uses: population.generated_people and
                population.generated_households
            marginal_variables (list(str)): list of marginal variables to compute error on.
        '''
        return Accuracy(
            person_pums=cleaned_data_persons.data,
            household_pums=cleaned_data_households.data,
            marginal_data=marginal_data.data,
            generated_persons=population.generated_people,
            generated_households=population.generated_households,
            marginal_variables=marginal_variables,
            use_all_marginals=use_all_marginals
        )

    @staticmethod
    def from_data_dir(state, puma, data_dir, marginal_variables, use_all_marginals):
        '''Helper method for loading datafiles with same format output by download_allocate_generate
        run script

        Args:
            state: state id
            puma: puma id
            data_dir: directory with stored csv files
            marginal_variables (list(str)): list of marginal variables to compute error on.

        Return: an initialized Accuracy object
        '''
        return Accuracy.from_csvs(
                state, puma,
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'persons_pums.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'households_pums.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'marginals.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'people.csv'),
                data_dir + os.path.sep + FILE_PATTERN.format(state, puma, 'households.csv'),
                marginal_variables,
                use_all_marginals
            )

    @staticmethod
    def from_csvs(
                state, puma,
                person_pums_filepath,
                household_pums_filepath,
                marginals_filepath,
                generated_persons_filepath,
                generated_households_filepath,
                marginal_variables,
                use_all_marginals
            ):
        '''Load csv files for use in accuracy calcs'''

        msg = '''Accuracy's from_data_dir Assumes files of the form:
            state_{state_id}_puma_{puma_id}_X
        Where X is contained in the set:
            {persons_pums.csv, households_pums.csv, marginals.csv, people.csv, households.csv}
        '''
        try:
            df_person = pd.read_csv(person_pums_filepath)
            df_household = pd.read_csv(household_pums_filepath)
            df_marginal = pd.read_csv(marginals_filepath)
            df_gen_persons = pd.read_csv(generated_persons_filepath)
            df_gen_households = pd.read_csv(generated_households_filepath)
        except IOError as e:
            print('{}\n{}'.format(msg, str(e)))
            raise IOError
        return Accuracy(df_person, df_household, df_marginal,
                        df_gen_persons, df_gen_households, marginal_variables, use_all_marginals)

    @staticmethod
    def _comparison_dataframe(
            person_pums, household_pums, marginal_data, generated_persons, generated_households,
            marginal_variables=[], use_all_marginals=True):
        '''Creates the dataframe containing all values by all sources, one per column, for use in
        the error metric calls in the Accuracy class.
        E.g:
                                 pums    gen  marginal
            (num_people, 1)      8584   7877      8132
            (num_people, 3)      7142   6911      6121
            (num_people, 2)     12218  12481     13753
            (num_people, 4+)    11863  11683     10775
            (age, 0-17)         32863  31368     30130
            (age, 18-34)        22476  18070     21490
            (age, 65+)          13997  12662     12568
            (age, 35-64)        45626  44232     45501

        Args:
            marginal_variables (list(str)): list of marginal variables to compute error on.
            use_all_marginals (bool): compute error on all eligible marginal variables.
                TAKES PRECEDENCE over the marginal_variables argument.
        Returns:
            dataframe with pums, marginal, and generated counts per variable
        '''

        variables = dict()
        if use_all_marginals is True and len(marginal_variables) <= 0:
            for control, bin_dict in marginals.CONTROLS.items():
                variables[control] = list(bin_dict.keys())  # cast for python3 compatibility
        else:
            for var in marginal_variables:
                variables[var] = list(marginals.CONTROLS[var].keys())

        # count should be its own marginal variable
        if 'num_people' in variables.keys():
            variables['num_people'].remove('count')

        comparison = OrderedDict()
        for variable, bins in variables.items():
            for bin in bins:
                comparison[(variable, bin)] = list()
                if variable == 'age':
                    comparison[(variable, bin)].append(
                        person_pums[person_pums[variable] == bin].person_weight.sum())
                    comparison[(variable, bin)].append(
                        generated_persons[generated_persons[variable] == bin].count()[0]
                    )
                elif variable == 'num_people':
                    comparison[(variable, bin)].append(household_pums[
                        household_pums[variable] == bin].household_weight.sum())
                    comparison[(variable, bin)].append(
                        generated_households[generated_households[variable] == bin].count()[0]
                    )
                elif variable == 'num_vehicles':
                    comparison[(variable, bin)].append(household_pums[
                        household_pums[variable] == bin].household_weight.sum())
                    comparison[(variable, bin)].append(
                        generated_households[generated_households[variable] == bin].count()[0]
                    )
                comparison[(variable, bin)].append(marginal_data[variable+'_'+bin].sum())
            # end bin
        # end variable
        return pd.DataFrame(list(comparison.values()), index=comparison.keys(),
                            columns=['pums', 'gen', 'marginal'])

    def root_mean_squared_error(self):
        '''Root mean squared error of the pums-marginals and generated-marginals vectors.
        No verbose option available due to the mean as an inner operation.
        Please use mean_root_squared_error for a verbose analog

        Returns:
            Two scalars. The first is root mean squared error of marginals to the raw micros, the
            second is root mean squared error of the marginals to the generated micros.
        '''
        df = self.comparison_dataframe
        return (
                np.sqrt(np.mean(np.square(df.pums - df.marginal))),
                np.sqrt(np.mean(np.square(df.gen - df.marginal)))
            )

    def root_squared_error(self):
        '''Similar to root mean squared error, but without the mean, in order to examine individual
        variables.

        Returns:
            Two vectors. The first is the root squared error of marginals to the raw micros, the
            second is the root squared error of the marginals to the generated micros.
        '''
        df = self.comparison_dataframe
        baseline = np.sqrt(np.square(df.pums - df.marginal))
        doppel = np.sqrt(np.square(df.gen - df.marginal))

        df = pd.DataFrame([baseline, doppel]).transpose()
        df.columns = ['marginal-pums', 'marginal-doppelganger']
        return df

    def absolute_pct_error(self):
        '''Accuracy in Absolute %Error. Mean is left out to avoid collapsing over individual
        variables.

        Returns:
            Two vectors. The first is A%E of marginals to the raw micros, the second is A%E of the
            marginals to the generated micros.
        '''
        df = self.comparison_dataframe
        baseline = np.abs(df.pums - df.marginal)/((df.pums + df.marginal)/2)
        doppel = np.abs(df.gen - df.marginal)/((df.gen + df.marginal)/2)

        df = pd.DataFrame([baseline, doppel]).transpose()
        df.columns = ['marginal-pums', 'marginal-doppelganger']
        return df

    @staticmethod
    def error_report(state_puma, data_dir, marginal_variables=[], use_all_marginals=True,
                     statistic=ErrorStat.ABSOLUTE_PCT_ERROR, verbose=False):
        '''Helper method to run an accuracy stats for multiple pumas
        Args:
            state_puma (dict): dictionary of state to puma list within the state.
            data_dir (str): directory with stored data in the form put out by
                download_allocate_generate
            E.g. load 3 Kansan (20) pumas and 2 in Missouri (29)
                state_puma['20'] = ['00500', '00602', '00604']
                state_puma['29'] = ['00901', '00902']
            variables (list(str)): vars to run error on. must be defined in marginals.py
            statistic (accuracy.ErrorStat): must be an implemented error statistic. See the
                ErrorStat class for a list of implemented error statistics.
            verbose (boolean): display per-variable error (inert for root mean squared error b/c of
                the inner mean)
        Returns:
            3 dataframes. Each contains two columns titled "x-y" with the error between sources
                x and y:
            error by puma (dataframe): marginal-pums, marginal-generated
            error by variable (dataframe): marginal-pums, marginal-generated
            mean error (dataframe): marginal-pums, marginal-generated
        '''
        diff_marginal_pums = OrderedDict()  # dictionary of marginal to pums differences
        diff_marginal_doppelganger = OrderedDict()  # dictionary of marginal to generated diffs
        for state, pumas in state_puma.items():
            for puma in pumas:
                if verbose:
                    print(' '.join(['\nrun accuracy:', state, puma]))
                accuracy = Accuracy.from_data_dir(state, puma, data_dir, marginal_variables)
                if statistic == ErrorStat.ABSOLUTE_PCT_ERROR:
                    diff_marginal_pums[(state, puma)] = \
                        accuracy.absolute_pct_error()['marginal-pums']
                    diff_marginal_doppelganger[(state, puma)] = \
                        accuracy.absolute_pct_error()['marginal-doppelganger']
                elif statistic == ErrorStat.ROOT_SQUARED_ERROR:
                    diff_marginal_pums[(state, puma)] = \
                        accuracy.root_squared_error()['marginal-pums']
                    diff_marginal_doppelganger[(state, puma)] = \
                        accuracy.root_squared_error()['marginal-doppelganger']
                else:
                    raise AccuracyException('Accuracy statistic not recognized. See the ErrorStat\
                         class for a list of implemented error statistics.')

        df_marginal_pums = pd.DataFrame(list(diff_marginal_pums.values()),
                                        index=diff_marginal_pums.keys())
        df_marginal_doppelganger = pd.DataFrame(list(diff_marginal_doppelganger.values()),
                                                index=diff_marginal_doppelganger.keys())

        df_by_puma = pd.DataFrame(
            [df_marginal_pums.mean(axis=1), df_marginal_doppelganger.mean(axis=1)],
            index=['marginal-pums', 'marginal-doppelganger']).transpose()

        df_by_var = pd.DataFrame(
                [df_marginal_pums.mean(axis=0), df_marginal_doppelganger.mean(axis=0)],
                index=['marginal-pums', 'marginal-doppelganger']
            ).transpose()
        if verbose:
            print('\nError by PUMA')
            print(df_by_puma.to_string())
            print('\n\nError by Variable')
            print(df_by_var.to_string())
            print('\nAverage: by PUMA, by Variable')
            print(df_by_var.mean().to_string())

        return df_by_puma, df_by_var, df_by_var.mean()
