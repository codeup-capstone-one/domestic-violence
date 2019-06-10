# ===========
# ENVIRONMENT
# ===========

import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator


def get_significant_t_tests(df, continuous_vars, target):
    '''
    Runs t-tests between two groups from 
    df: a dataframe and 
    continuous_vars: a list of column names.
    against
    target: a comparative variable from df to test between

    If test results are noteworthy due to the t-statistic and p-value, results are printed

    Returns a list of perceived significant features and a dictionary with variable name and t-statistic
    '''
    some_feats = []
    some_dict = {}
    train_df = df
    buildstr1 = r'train_df[train_df.' + target + r' == 1].'
    buildstr2 = r'train_df[train_df.' + target + r' == 0].'
    for feat in continuous_vars:
        affirmative = buildstr1 + feat
        negative = buildstr2 + feat
        tstat, pval = stats.ttest_ind(eval(affirmative), eval(negative))
        if tstat > 1.96 or tstat < -1.96:
            if pval < 0.05:
                print(f'Feature analyzed: {feat}')
                print(
                    'Our t-statistic is {:.4} and the p-value is {:.10}'.format(tstat, pval))
                print('----------')
                some_feats.append(feat)
                some_dict[feat] = tstat
    return some_feats, some_dict


def get_chi_squared(df, features, target):
    '''
    Runs chi-squared test between two categorical variables from 
    df: a dataframe and
    features: a list of columns
    target: a variable name to compare to the cycled features list

    Results are printed depending on perceived significance
    Returns a list of significant features as well as a dictionary with chi-stat
     '''
    train_df = df
    sig_feats = []
    sig_dict = {}
    for feat in features:
        tbl = pd.crosstab(train_df[feat], train_df[target])
        stat, p, dof, expected = stats.chi2_contingency(tbl)
        prob = .95
        critical = stats.chi2.ppf(prob, dof)
        if abs(stat) >= critical:
            if p < 0.05:
                sig_dict[feat] = abs(stat)
                print(feat)
                print('Dependent (reject H0)')
                print('-----------------------')
                sig_feats.append(feat)
        else:
            pass
    return sig_feats, sig_dict


def sort_sigs(a_dict):
    '''
    takes in a dictionary (a_dict) and sorts based on values associated with each key
    '''
    val_list = []
    for key in a_dict:
        val_list.append(a_dict[key])
        sorted_vals = sorted(
            a_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vals


def combine_significants(dict1, dict2):
    '''
    takes in 
    dict1 and dict2: two dictionaries
    converts to two lists of tuples and creates 
    a new list with the 
    first element of each of the argument lists
    returns a new list of all features
    '''
    list1 = sort_sigs(dict1)
    list2 = sort_sigs(dict2)
    new_list = []
    for item in list1:
        new_list.append(item[0])
    for item in list2:
        if item not in(new_list):
            new_list.append(item[0])
    return new_list


def swarrrm(df, cat, num_vars):
    '''
    creates a series of swarm plots from a dataframe (df)
     using a categorical variable (cat) 
     and a list of continuous ones (num_vars)
     '''
    for i, col in enumerate(num_vars):
        i = i+1
        plt.figure(figsize=(len(num_vars), len(num_vars)*2))
        plt.subplot(len(num_vars), 1, i)
        sns.swarmplot(data=df, x=cat, y=col)
        plt.show()


def make_rel(df, x, y, hue):
    '''
    creates a relplot from a dataframe df using 
    two continuous (x, y)
    and one categorical (hue) variable
    '''
    sns.relplot(x=x, y=y, hue=hue, data=df)


def make_rels(df, feat1, feat2, features):
    '''
    takes in a a dataframe df,
    feat1 and fat2:  two feature variable names, and 
    features: a list of other features for comparative relplots
    '''
    for feat in features:
        make_rel(df, feat1, feat2, feat)
        plt.show()


def make_bars(df, metric, features):
    '''
    creates bar plots based on
    df: a pandas dataframe,
    metric: a string literal for the rate being evaluated and
    features: a list of categorical variables 
    '''
    _, ax = plt.subplots(nrows=len(features), ncols=1,
                         figsize=(15, (11*len(features))))
    train_df = df
    buildstr1 = r'train_df.' + metric + r'.mean()'
    rate = eval(buildstr1)
    status = metric + '_status'
    for i, feature in enumerate(features):
        sns.barplot(feature, metric, data=train_df, ax=ax[i], alpha=.5)
        ax[i].set_ylabel(metric)
        ax[i].axhline(rate, ls='--', color='grey')


def plot_hist(df):
    """
    Plots the distribution of the dataframe's variables.
    """
    df.hist(figsize=(24, 20), bins=20)
