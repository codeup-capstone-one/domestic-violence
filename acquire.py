#!/usr/bin/env python

"""
This script contains code used by the following jupytr notebooks:

1. dd-wrangling.ipynb
2. env.py
3.

"""


# ===========
# ENVIRONMENT
# ===========


import os
import sys

import pandas as pd

from env import path


# variables to use by selection of features from data dictionary:


ONE_COLS = ['CASEID',
            'ABUSED',
            'SCRSTATR',
            'LENGTHC1',
            'C1SITUAT',
            'PABUSE',
            ]


TWO_COLS = ['CASEID',
            'D3RCHILT',
            ]

THREE_COLS = ['CASEID',
              'E13PRGNT',
              'N7PREGNT',
              'TOTSUPRT'
              ]

FOUR_COLS = ['CASEID',
             'G1NUMBER',
             'H1JEALUS',
             'H2LIMIT',
             'H3KNOWNG',
             'J1HIT',
             'J2THROWN',
             'J3PUSH',
             'J4SLAP',
             'J5KICK',
             'J6OBJECT',
             'J7BEAT',
             'J8CHOKE',
             'J9KNIFE',
             'J10GUN',
             'J11SEX',
             'POWER',
             'HARASS',
             ]

FIVE_COLS = ['CASEID',
             'B1AGE',
             'AGEDISP',
             'STDETAI',
             ]

SEVEN_COLS = ['CASEID',
              'SAMESEXR',
              'N11DRUGS',
              'N12ALCHL',
              'N13SUHIM',
              'N16CHILD',
              'N17ARRST',
              'N1FRQNCY',
              'N2SVRITY',
              'N3WEAPON',
              'N4CHOKE',
              'N5SEX',
              'N6CONTRL',
              'N8JEALUS',
              'N10CPBLE',
              ]

SIX_COLS = ['CASEID',
            'M5FIRED',
            'M11HIGH',
            'M35SAFE',
            'M41ILLGL',
            'M42DAGRR',
            'M13TALKR',
            'M32OTHER',
            'M27HOW',
            'M30ARRES',
            'M31HOW',
            'M38ORDER',
            ]

ELEVEN_COLS = ['CASEID',
               'SEVERER',
               'TOTINCR',
               'THREATR',
               'SLAPR',
               'PUNCHR',
               'BEATR',
               'UWEAPON',
               'FORCEDR',
               'MISCARR',
               'RESTRAIN',
               'CHOKED',
               'NDRUNK',
               'RDRUNK',
               'BOTHDRUN',
               'NDRUGS',
               'RDRUGS',
               'BOTHDRUG',
               ]

# ===========
# ACQUISITION
# ===========


def read_data(filename):
    """
    Reads in a subset of the dataset: Chicago Women's
    Health Risk Study, 1995-1998 (ICPSR 3002)

    """
    return pd.read_csv(path + filename, low_memory=False)


def join_data(filename1, filename2):
    """
    Merges the two subsets into one dataset.

    """
    df1 = read_data(filename1)
    df2 = read_data(filename2)
    return pd.merge(df1, df2, on='CASEID', how='inner')


def get_data():
    df1 = read_data('data01.csv')
    df2 = read_data('data02.csv')
    df3 = read_data('data03.csv')
    df4 = read_data('data04.csv')
    df5 = read_data('data05.csv')
    df6 = read_data('data06.csv')
    df7 = read_data('data07.csv')
    df11 = read_data('data11.csv')

    df1 = df1[ONE_COLS]
    df2 = df2[TWO_COLS]
    df3 = df3[THREE_COLS]
    df4 = df4[FOUR_COLS]
    df5 = df5[FIVE_COLS]
    df7 = df7[SEVEN_COLS]

    dfa = df1.merge(right=df2, on='CASEID')
    dfa = dfa.merge(right=df3, on='CASEID')
    dfa = dfa.merge(right=df4, on='CASEID')
    dfa = dfa.merge(right=df5, on='CASEID')
    dfa = dfa.merge(right=df7, on='CASEID')

    df6 = df6[SIX_COLS]
    df11 = df11[ELEVEN_COLS]

    dfb = df6.merge(right=df11, on='CASEID')

    return dfa, dfb
# ==================================================
# MAIN
# ==================================================


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    """Main entry point for the script."""
    pass


if __name__ == '__main__':
    sys.exit(main())


__authors__ = ["Ednalyn C. De Dios", "Jesse Ruiz", "Matthew Capper"]
__copyright__ = "Copyright 2019, Codeup Data Science"
__credits__ = ["Maggie Guist", "Zach Gulde"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainers__ = "Ednalyn C. De Dios"
__email__ = "ednalyn.dedios@gmail.com"
__status__ = "Prototype"
