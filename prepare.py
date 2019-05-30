import pandas as pd
import numpy as np
import acquire

df10 = acquire.read_data('data10.csv')


def make_repeat_series(df10):
    '''takes a dataframe with a caseid columns and returns a series with offense numbers using a groupby'''
    repeat_series = df10.groupby('CASEID').INCIDENT.count()
    return repeat_series


def over_1(repeat_series):
    '''takes a pandas series and tests for a value to put in a list of caseIDs that are repeat offenses'''
    repeat_cases = []
    for case, inc_num in enumerate(repeat_series):
        if inc_num > 1:
            repeat_cases.append(repeat_series.index[case])
    return repeat_cases


def get_repeat_case(val):
    '''takes a value and establishes if it meets criteria to be in repeat offenses'''
    repeat_cases = over_1(make_repeat_series(df10))
    if val in repeat_cases:
        return 1
    else:
        return 0


def value_counts(dataframe):
    ''' assesses that column is not the primary key and presents values'''
    for col in dataframe:
        print(col)
        if col in(['CASEID', 'id']):
            pass
        elif np.issubdtype(dataframe[col].dtype, np.number) and dataframe[col].nunique() > 20:
            print(dataframe[col].value_counts(bins=10, sort=False))
        else:
            print(dataframe[col].value_counts(sort=False))
        print('\n-------------------------------------------------------------\n')


def rename_columns(df):
    '''takes in selected dataframe and renames columns to intuitive non-capitalized titles'''
    df.rename(columns={'CASEID': 'id',
                       'ABUSED': 'abuse_past_year',
                       'SCRSTATR': 'abuse_status',
                       'LENGTHC1': 'length_relationship',
                       'C1SITUAT': 'partner_abusive',
                       'PABUSE': 'num_abusers',
                       'D3RCHILT': 'num_children',
                       'E13PRGNT': 'pregnant',
                       'N7PREGNT': 'beaten_while_pregnant',
                       'TOTSUPRT': 'support_score',
                       'G1NUMBER': 'guns_in_home',
                       'H1JEALUS': 'jealous_past_year',
                       'H2LIMIT': 'limit_family_contact',
                       'H3KNOWNG': 'location_tracking',
                       'J1HIT': 'threat_hit',
                       'J2THROWN': 'thrown_object',
                       'J3PUSH': 'push_shove',
                       'J4SLAP': 'slap',
                       'J5KICK': 'kick_punch',
                       'J6OBJECT': 'hit_object',
                       'J7BEAT': 'beaten',
                       'J8CHOKE': 'choked',
                       'J9KNIFE': 'threat_knife',
                       'J10GUN': 'threat_gun',
                       'J11SEX': 'rape_with_threat',
                       'POWER': 'power_scale',
                       'HARASS': 'harass_scale',
                       'B1AGE': 'id_age',
                       'AGEDISP': 'age_disparity',
                       'STDETAI': 'children_not_partner',
                       'SAMESEXR': 'same_sex_relationship',
                       'N11DRUGS': 'partner_drug_use',
                       'N12ALCHL': 'partner_alcohol_use',
                       'N13SUHIM': 'threat_suicide',
                       'N16CHILD': 'partner_reported_child_abuse',
                       'N17ARRST': 'partner_arrested',
                       'N1FRQNCY': 'violence_increased',
                       'N2SVRITY': 'severity_increased',
                       'N3WEAPON': 'weapon_ever',
                       'N4CHOKE': 'choked_ever',
                       'N5SEX': 'rape_ever',
                       'N6CONTRL': 'controlled_ever',
                       'N8JEALUS': 'jealous',
                       'N10CPBLE': 'capable_murder',
                       'RECID': 'reassault'
                       }, inplace=True)


def replace_nonvals(df):
    '''assesses values in column of a dataframe are in numerical format and replaces
    any missing values as per our data dictionary with an imputed zero value.'''
    for col in df:
        if col in(['CASEID', 'id']):
            pass
        elif col not in(['length_relationship',
                         'num_abusers',
                         'num_children',
                         'power_scale',
                         'harass_scale',
                         'id_age',
                         'age_disparity',
                         'children_not_partner']):
            df[col].replace([2, 3, 9, 555, 666, 777, 888,
                             999, 9999], 0, inplace=True)
            df[col].replace(4, 1, inplace=True)
        if col == 'guns_in_home':
            df[col].replace([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 1, inplace=True)
        elif col == 'num_children':
            df[col].replace(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2, inplace=True)
        elif col == 'num_abusers':
            df[col].replace([2, 3], 2, inplace=True)
            df[col].replace([9, 0], inplace=True)
        elif col == 'beaten_while_pregnant':
            df[col].replace(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 0, inplace=True)
        elif col == 'age_disparity':
            df[col].replace([1, 999], 0, inplace=True)
            df[col].replace(2, 1, inplace=True)
            df[col].replace(3, 2, inplace=True)
            df[col].replace(4, -1, inplace=True)
            df[col].replace(5, -2, inplace=True)
            df[col].replace(6, -3, inplace=True)

# dfb['RECID'] = dfb.CASEID.apply(get_repeat_case)
