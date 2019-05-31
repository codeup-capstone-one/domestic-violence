import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator


def get_significant_t_tests(df, continuous_vars):
    '''Runs t-tests between two groups from a dataframe and a list of column names.
    If test results are noteworthy due to the t-statistic and p-value, results are printed
    Returns a list of perceived significant features and a dictionary with '''
    some_feats = []
    some_dict = {}
    train_df = df
    buildstr1 = r'train_df[train_df.abuse_past_year == 1].'
    buildstr2 = r'train_df[train_df.abuse_past_year == 0].'
    for feat in continuous_vars:
        abused = buildstr1 + feat
        not_abused = buildstr2 + feat
        tstat, pval = stats.ttest_ind(eval(abused), eval(not_abused))
        if tstat > 1.96 or tstat < -1.96:
            if pval < 0.05:
                print(f'Feature analyzed: {feat}')
                print('Comparing abused to non-abused: ')
                print(
                    'Our t-statistic is {:.4} and the p-value is {:.10}'.format(tstat, pval))
                print('----------')
                some_feats.append(feat)
                some_dict[feat] = tstat
    return some_feats, some_dict


def get_chi_squared(df, features):
    '''runs chi-squared test between two categorical variables from a dataframe (df) and
     list of columns (features) results are printed depending on perceived significance
     returns a list of significant features as well as a dictionary with chi-stat'''
    train_df = df
    sig_feats = []
    sig_dict = {}
    for feat in features:
        tbl = pd.crosstab(train_df[feat], train_df['abuse_past_year'])
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
    val_list = []
    for key in a_dict:
        val_list.append(a_dict[key])
        sorted_vals = sorted(
            a_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vals

def combine_significants(dict1, dict2):
    '''takes in two lists of tuples and creates a new list with the 
    first element of each of the argument lists
    returns a new list of all features'''
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
    '''creates a series of swarm plots from a dataframe using a categorical variable and a list of continuous ones'''
    for i, col in enumerate(num_vars):
        i = i+1
        plt.figure(figsize=(len(num_vars)*2, 14))
        plt.subplot(len(num_vars), 1, i)
        sns.swarmplot(data=df, x=cat, y=col)
        plt.show()


def make_rel(df, x, y, hue):
    '''creates a relplot from a dataframe using two continuous and one categorical variable as hue'''
    sns.relplot(x=x, y=y, hue=hue, data=df)


def make_rels(df, feat1, feat2, features):
    '''takes in a a dataframe, two features, and a list of other features for comparative relplots'''
    for feat in features:
        make_rel(df, feat1, feat2, feat)
        plt.show()


def make_bars(df, metric, features):
    '''creates bar plots for categorical variables specified in features parameter as a list,
    metric as a string literal for the rate being evaluated,
    and df as the greater dataframe being utilized'''
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
