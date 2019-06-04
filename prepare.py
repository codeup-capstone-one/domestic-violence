# ===========
# ENVIRONMENT
# ===========

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


def rename_columns_all(df):
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


def replace_nonvals_all(df):
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
            # initial weed-out maps 2 to zero as a 'no' response
            # maps 3, 9, and unreliable/error codes due to unreliable or out of scope responses
            df[col].replace([2, 3, 9, 555, 666, 777, 888,
                             999, 9999], 0, inplace=True)
            # maps 4 to an affirmative 1 response. correlates to 'yes but not in past year'
            df[col].replace(4, 1, inplace=True)
        if col == 'guns_in_home':
            # alters response of question to an affirmative binary if more than one gun in home
            df[col].replace([2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                             12, 13, 14, 15], 1, inplace=True)
        elif col == 'num_children':
            # bins 2+ children into one category of '2'
            df[col].replace(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2, inplace=True)
        elif col == 'num_abusers':
            # bins multiple abusers, maps unavailable info to zero
            df[col].replace([2, 3], 2, inplace=True)
            df[col].replace(9, 0, inplace=True)
        elif col == 'beaten_while_pregnant':
            # remaps beaten while pregnant to a binary
            df[col].replace(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 0, inplace=True)
        elif col == 'age_disparity':
            # remaps responses to age disparity
            df[col].replace([1, 999], 0, inplace=True)
            df[col].replace(2, 1, inplace=True)
            df[col].replace(3, 2, inplace=True)
            df[col].replace(4, -1, inplace=True)
            df[col].replace(5, -2, inplace=True)
            df[col].replace(6, -3, inplace=True)
        elif col == 'power_scale':
            df[col].replace(999, 0, inplace=True)


def get_nulls_by_column(df):
    '''gives analysis of dataframe and prints out nulls by column'''
    sum_nulls_col = df.isna().sum()
    percent_nulls_col = df.isna().sum()/len(df.columns)
    nulls_by_col = pd.concat([sum_nulls_col, percent_nulls_col], names=[
                             'sum_nulls_col', 'percent_nulls_col'], axis=1)
    nulls_by_col['sum_nulls'] = nulls_by_col[0]
    nulls_by_col['nulls_by_percent'] = nulls_by_col[1]
    nulls_by_col.drop(columns=[0, 1], inplace=True)
    nulls_by_col = nulls_by_col.loc[~(nulls_by_col == 0).all(axis=1)]
    print(nulls_by_col)


def get_nulls_by_row(df):
    '''gives analysis of dataframe and prints out nulls by row'''
    df.reset_index(inplace=True, drop=True)
    rows = len(df.index)
    nulls_by_row = pd.DataFrame
    for ind in range(rows):
        null_vals = df.loc[ind].isna().sum()
        percent = (null_vals/(len(df.loc[ind]))*100)
        if null_vals > 0:
            print('row: {} count nulls: {}, percent nulls in row: {:.2f}.'.format(
                ind, null_vals, percent))


def handle_missing_threshold(df, prop_required_column=.3, prop_required_row=.9):
    '''removes na values from dataframe based on inputted threshold value'''
    threshold = int(round(prop_required_column*len(df.index), 0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns), 0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def summarize_data(df):
    '''prints out dataframe head, tail, shape, info and value counts'''
    df_head = df.head()
    print(f'HEAD\n{df_head}', end='\n\n')

    df_tail = df.tail()
    print(f'TAIL\n{df_tail}', end='\n\n')

    shape_tuple = df.shape
    print(f'SHAPE: {shape_tuple}', end='\n\n')

    df_describe = df.describe()
    print(f'DESCRIPTION\n{df_describe}', end='\n\n')

    df.info()
    print(f'INFORMATION')

    value_counts(df)

# brief note reminder on creating recidivism column:
# dfb['RECID'] = dfb.CASEID.apply(get_repeat_case)



def rename_columns_recid(df):
    '''takes in selected dataframe and renames columns to intuitive non-capitalized titles'''
    df.rename(columns={'CASEID': 'id',
                        'M5FIRED' : 'gun_fired',
                       'M11HIGH': 'anyone_high', 
                       'M35SAFE' : 'safe_place', 
                       'M41ILLGL' : 'forced_illegal', 
                       'M42DAGRR' : 'life_danger',
                        'M13TALKR' : 'talk_about_it', 
                        'M32OTHER' : 'left_or_not',
                        'M27HOW': 'medical_staff_helpful',
                        'M30ARRES' : 'perp_arrested_ever', 
                        'M31HOW' : 'police_resp', 
                        'M38ORDER' : 'order_protection',
                        'SEVERER' : 'level_severity', 
                        'TOTINCR' : 'num_incidents', 
                        'THREATR' : 'num_threats', 
                        'SLAPR' : 'num_slapping', 
                        'PUNCHR' : 'num_punching', 
                        'BEATR' : 'num_beating', 
                        'UWEAPON' : 'num_weapon',
                        'FORCEDR' : 'num_forced_sex', 
                        'MISCARR' : 'miscarriage_resulted', 
                        'RESTRAIN' : 'restrained_by_perp', 
                        'CHOKED' : 'num_choked', 
                        'NDRUNK' : 'num_perp_drunk',
                        'RDRUNK' : 'num_woman_drunk',
                        'BOTHDRUN' : 'num_both_drunk', 
                        'NDRUGS' : 'num_perp_drugs', 
                        'RDRUGS' : 'num_woman_drugs', 
                        'BOTHDRUG' : 'num_both_drugs', 
                       'RECID': 'reassault'
                       }, inplace=True)

def replace_nonvals_recid(df):
    '''assesses values in column of dataframe with reassault cases are in numerical format and replaces
    any missing values as per our data dictionary with an imputed zero value.'''
    for col in df:
        if col in(['CASEID', 'id']):
            pass
        elif col in(['gun_fired',
                        'anyone_high',
                         'safe_place',
                         'forced_illegal',
                         'life_danger',
                         'talk_about_it']):
            # initial weed-out maps 2 to zero as a 'no' response
            # maps 888, 999, 9999 and unreliable/error codes due to unreliable or out of scope responses
            df[col].replace([2, 888, 999, 9999], 0, inplace=True)
        if col == 'order_protection':
            df[col].replace([2, 3, 999, 9999], 0, inplace=True)
        elif col == 'left_or_not':
            df[col].replace([11,12,13,14,15,16,17,18,19], 1, inplace=True)
            df[col].replace([21,22,31,32,33,41,42,43,44,45,46,99], 0, inplace=True)

        elif col == 'medical_staff_helpful':
            df[col].replace([7777, 99999, 9999], 0, inplace=True)

        # elif col == 'perp_arrested_ever':
        #     df[col].replace(2, 'removed', inplace=True)
        #     df[col].replace(3, 0, inplace=True)
        #     df[col].replace(999, 'missing', inplace=True)
        # elif col == 'police_resp':
        #     df[col].replace(777, 'na', inplace=True)
        #     df[col].replace(999, 'missing', inplace=True)
        # elif col == 'order_protection':
        #     df[col].replace([2, 3], 0, inplace=True)
        #     df[col].replace([999, 9999], 'missing/unknown', inplace=True)
        # elif col == 'level_severity':
        #     df[col].replace(9, 'missing', inplace=True)
        # elif col == 'num_incidents':
        #     df[col].replace(999, 'missing', inplace=True)
        # elif col == 'num_threats':
        #     df[col].replace(999, 'missing', inplace=True)
        # elif col == 'num_slapping':
        #     df[col].replace(999, 'missing', inplace=True)
