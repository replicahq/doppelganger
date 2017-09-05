# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import pandas

from doppelganger import inputs


class DataSource(object):

    @classmethod
    def from_csv(cls, infile):
        raise NotImplementedError()

    def write(self, outfile):
        self.data.to_csv(outfile)


class DirtyDataSource(DataSource):

    def __init__(self, data, name_map):
        self.data = data
        self.name_map = name_map

    def clean(self, field_names, preprocessor, state=None, puma=None):
        cleaned_data = preprocessor.process_dataframe(self.data, field_names, self.name_map)
        if puma is not None:
            cleaned_data = cleaned_data[
                    (cleaned_data[inputs.STATE.name].astype(str) == str(state)) &
                    (cleaned_data[inputs.PUMA.name].astype(str) == str(puma))
                ]
        return CleanedData(cleaned_data)


class PumsData(DirtyDataSource):

    HOUSEHOLD_TABLE = 'households'
    PERSONS_TABLE = 'persons'

    def __init__(self, data):
        name_map = {field.name: field.pums_name for field in inputs.PUMS_INPUTS}
        return super(PumsData, self).__init__(data, name_map)

    @staticmethod
    def from_database(conn, state_id, puma_id, schema_name, table_name, fields):
        columns = ', '.join(field.pums_name for field in fields)
        query = ('''
            SELECT {columns}
            FROM {schema}.{table}
            WHERE ST=\'{state_id}\' AND PUMA=\'{puma_id}\'
            ORDER BY SERIALNO
            ;''').format(
                columns=columns,
                schema=schema_name,
                table=table_name,
                state_id=state_id,
                puma_id=puma_id
            )
        return PumsData(pandas.read_sql_query(query, conn))

    @staticmethod
    def from_csv(infile):
        data = pandas.read_csv(infile)
        return PumsData(data)


class CleanedData(DataSource):

    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_csv(infile):
        data = pandas.read_csv(infile)
        return CleanedData(data)
