'''Download pums data from a db to speed up run-times.'''
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import psycopg2
from doppelganger import datasource, allocation, inputs, Preprocessor

HOUSEHOLD_TABLE = 'households'
PERSONS_TABLE = 'persons'


def fetch_pums_data(state_id, puma_id, configuration,
                    db_host, db_database, db_schema, db_user, db_password):
    '''Download PUMS data from pums tables stored in a database
    Args:
        state_id: 2-digit state fips code
        puma_id: 5-digit puma code
        configuration: contains person and household fields, along with how to instruct the
            preprocessor to discretize the fields
        db_host: hostname of the POSTGRESQL instance to connect to
        db_database: database name to connect to
        db_schema: schema which _must_ contain a person and household table with pums fields
            referenced in doppelganger/inputs.py
        db_user: username to connect with
        db_password: password to authenticate to the database

    Returns:
        person_data: a PumsData wrapped dataframe whose fields have been mapped according to
            inputs.py
        households_data: same as person_data but for households
    '''

    preprocessor = Preprocessor.from_config(configuration.preprocessing_config)

    # Union default and extra fields
    person_fields = link_fields_to_inputs(configuration.person_fields)
    person_fields = allocation.DEFAULT_PERSON_FIELDS.union(person_fields)
    person_fieldnames = tuple(set(p.name for p in person_fields))

    household_fields = link_fields_to_inputs(configuration.household_fields)
    household_fields = allocation.DEFAULT_HOUSEHOLD_FIELDS.union(household_fields)
    household_fieldnames = tuple(set(hh.name for hh in household_fields))

    puma_conn = None
    try:
        puma_conn = psycopg2.connect(
            host=db_host,
            database=db_database,
            user=db_user,
            password=db_password,
        )

        households_data = datasource.PumsData.from_database(
            conn=puma_conn,
            state_id=state_id,
            puma_id=puma_id,
            schema_name=db_schema,
            table_name=HOUSEHOLD_TABLE,
            fields=household_fields
        ).clean(
            field_names=household_fieldnames,
            preprocessor=preprocessor,
            state=state_id,
            puma=puma_id
        )

        persons_data = datasource.PumsData.from_database(
            conn=puma_conn,
            state_id=state_id,
            puma_id=puma_id,
            schema_name=db_schema,
            table_name=PERSONS_TABLE,
            fields=person_fields
        ).clean(
            field_names=person_fieldnames,
            preprocessor=preprocessor,
            state=state_id,
            puma=puma_id
        )

    except psycopg2.DatabaseError as error:
        print(error)
    finally:
        if puma_conn is not None:
            puma_conn.close()
            print('Database connection closed.')

    return households_data, persons_data


def link_fields_to_inputs(input_list):
    '''Verify extra field references have been properly registered in inputs.py
    Map them to their inputs data-types based on their name property.

    Args:
        input_list: list of variable names defined in doppelganger/inputs.py

    Returns:
        set of PUMS_INPUTS objects for use in PumsData.from_database
    '''
    input_name_map = {x.name: x for x in inputs.PUMS_INPUTS}
    if not all(x in input_name_map.keys() for x in input_list):
        raise ValueError('One or more extra fields not registered in doppelganger/inputs.py')
    return set([input_name_map[i] for i in input_list])
