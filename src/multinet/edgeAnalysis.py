#!/usr/bin/env python3

from multinet.utils import \
    Config, loadData, validateCols, checkDuplicates, \
    prepareData, extractCodeTimes
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.sparse import csr_matrix
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import StratifiedTable


def edgeAnalysis(config: str):
    config = Config(config).config
    df = loadData(config)
    # Retrieve full index (including records with no morbidities)
    fullIndex = getFullIndex(df)
    codePairs = getMMFrequency(df)
    df_long = getICDlong(df)
    df_sp = df2Sparse(df_long, fullIndex)
    allLinks = runEdgeAnalysis(df_sp, codePairs)
    return allLinks, config


def getFullIndex(df: pd.DataFrame) -> pd.Series:
    """ Retain original index to ensure
        all records are added to sparse matrix """
    return df.reset_index()[['index', 'strata']].apply(tuple, axis=1)


def getMMFrequency(df: pd.DataFrame) -> pd.Series:
    """Process ICD-10 multi-morbidity data.

    Args:
        df (pd.DataFrame): ICD-10 data processed by cma.processData().

    Returns:
         Mulimorbidity frequency of pairwise ICD-10 codes.
    """
    func = lambda x: [tuple(sorted(x)) for x in combinations(x, 2)]
    codePairs = (
        df['codes'].apply(func).explode().drop_duplicates().dropna().tolist()
    )
    return codePairs


def getICDlong(df: pd.DataFrame) -> pd.DataFrame:
    """Generate long-format ICD data, 1 code per row.

    Args:
        df (pd.DataFrame): ICD-10 data processed by cma.processData().

    Returns:
         Long format ICD-10 codes.
    """
    df_long = (
        df.reset_index()
        .rename({'index': 'ID'}, axis=1)
        .set_index(['ID', 'strata'])
        .apply(pd.Series.explode)
        .reset_index()
        .dropna()
    )
    if df.attrs['directed']:
        df_long['time'] = df_long['time'].astype(int)
    else:
        df_long['time'] = df_long['time'].astype(bool)
    return df_long


def df2Sparse(df: pd.DataFrame, fullIndex: pd.Series) -> pd.DataFrame:
    indexGrp = df.groupby(['ID', 'strata']).grouper
    indexIdx = indexGrp.group_info[0]
    colGroup = df.groupby(['codes']).grouper
    colIdx = colGroup.group_info[0]
    df_sparse = csr_matrix(
        (df['time'].values, (indexIdx, colIdx)),
        shape=(indexGrp.ngroups, colGroup.ngroups)
    )
    df_sparse = (
        pd.DataFrame.sparse.from_spmatrix(
            df_sparse, index=list(indexGrp),
            columns=list(colGroup))
        .reindex(fullIndex)
        .fillna(0)
    )
    # Move stata to column
    df_sparse['strata'] = df_sparse.index.map(lambda x: x[1]).values
    # Set index as ID
    df_sparse.index = df_sparse.index.map(lambda x: x[0]).values
    return df_sparse


def runEdgeAnalysis(df_sp, codePairs):
    indices = retrieveIndices(df_sp)
    allLinks = []
    for m1, m2 in codePairs:
        a1 = np.array(df_sp[m1])
        a2 = np.array(df_sp[m2])
        z, p, total = proportionTest(a1, a2)
        k, minObs = stratifiedOdds(a1, a2, indices)
        if k is None:
            continue
        allLinks.append([
            m1, m2, minObs, k.oddsratio_pooled, k.riskratio_pooled,
            k.test_null_odds().pvalue, z, p, total
        ])
    allLinks = pd.DataFrame(allLinks)
    allLinks.columns = ([
        'Node1', 'Node2', 'minObs', 'OR', 'RR',
        'pNull', 'zProp', 'pProp', 'totalPositive'
    ])
    # FDR correct all valid tests
    allLinks.loc[allLinks['pNull'].notna(), 'FDRnull'] = (
        fdrcorrection(allLinks.loc[allLinks['pNull'].notna(), 'pNull'])[1]
    )
    allLinks.loc[allLinks['pProp'].notna(), 'FDRprop'] = (
        fdrcorrection(allLinks.loc[allLinks['pProp'].notna(), 'pProp'])[1]
    )
    # Larger odds ratio = smaller edge
    allLinks['inverseOR'] = 1 / allLinks['OR']
    allLinks['inverseRR'] = 1 / allLinks['RR']

    return allLinks


def retrieveIndices(df: pd.DataFrame) -> list:
    """ Retrieve per-strata indices """
    indices = []
    for stratum in df['strata'].unique():
        indices.append(np.array(df['strata'] == stratum))
    return indices


def proportionTest(a1, a2):
    bothTimeInfo = (a1 > 0) & (a2 > 0)
    a1_to_a2 = (a1[bothTimeInfo] < a2[bothTimeInfo]).sum()
    a2_to_a1 = (a2[bothTimeInfo] < a1[bothTimeInfo]).sum()
    z, p = proportions_ztest(
        a1_to_a2, a1_to_a2 + a2_to_a1, value=0.5, alternative='two-sided')
    # Total positive cases (ignoring time information)
    total = ((a1 != 0) & (a2 != 0)).sum()
    return z, p, total


def stratifiedOdds(a1, a2, indices):
    tables = makeStratifiedTable(a1, a2, indices)
    if not tables:
        return (None, None)
    minObs = np.sum(tables, axis=2).min()
    k = StratifiedTable(tables)
    return k, minObs


def makeStratifiedTable(
    a1: np.array,
    a2: np.array,
    strataIndices: np.array,
) -> np.array:
    """ Generate set of stratified contigency tables """
    ctTables = []
    a1 = a1.copy()
    a2 = a2.copy()
    for s in strataIndices:
        ct = np.bincount(
            2 * a1[s].astype(bool) + a2[s].astype(bool),
            minlength=4).reshape(2,2)
        if (ct == 0).any():
            continue
        ctTables.append(ct)
    return ctTables
