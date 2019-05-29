import pandas as pd

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

# dfb['RECID'] = dfb.CASEID.apply(get_repeat_case)