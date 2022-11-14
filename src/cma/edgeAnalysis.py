#!/usr/bin/env python3

from cma.utils import Config
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


def loadData(config: dict, keepStrata: bool = False) -> pd.DataFrame:
    """ Main function for loading data """
    # Enfore codes as strings
    dtypes = {col: str for col in config['codeCols']}
    if config['seperator'] is None:
        data = pd.read_csv(
            config['input'], chunksize=config['chunkSize'],
            dtype=dtypes, iterator=True
        )
    else:
        data = pd.read_csv(
            config['input'], sep=config['seperator'],
            chunksize=config['chunkSize'], dtype=dtypes, iterator=True
        )
    allData = []
    rowsWithDups = []
    for i, chunk in enumerate(data):
        if i == 0:
            # Ensure all column names are in df
            validateCols(chunk, config)
        # Check for duplicate names
        checkDuplicates(chunk, config)
        allData.append(prepareData(chunk, config, keepStrata))
    allData = pd.concat(allData).fillna('')
    allData.attrs['directed'] = config['directed']
    return allData


def validateCols(df, config):
    """ Check for missing columns in df """
    missingCols = set(config['allCols']) - set(df.columns)
    if missingCols:
        logging.error(f'{missingCols} not present in {config["input"]}\n')
        raise ValueError
    if config['directed']:
        timeTypes = df[config['timeCols']].select_dtypes(
            include=[np.number, np.datetime64])
        invalidType = set(config['timeCols']) - set(timeTypes.columns)
        if invalidType:
            logging.error(
                f'Invalid time type at columns {invalidType} in {config["input"]}\n')
            raise ValueError


def checkDuplicates(df, config):
    """ Check for duplicate codes in row """
    duplicates = df[config['codeCols']].apply(
        lambda x: len(set(x.dropna())) < len(x.dropna()), axis=1)
    return duplicates[duplicates].index


def prepareData(df: pd.DataFrame, config: dict, keepStrata: bool = False) -> pd.DataFrame:
    """Process ICD-10 multi-morbidity data.

    Args:
        df (pd.DataFrame) : ICD-10 Data.
        config (str) : Preloaded config file.

    Returns:
        Processed DataFrame of strata and ICD-10 codes.
    """
    args = (config['codeCols'], config['timeCols'])
    df[['codes', 'time']] = df.apply(extractCodeTimes, args=args, axis=1)
    if config['strata']:
        df['strata'] = df[config['strata']].apply(tuple, axis=1)
    else:
        df['strata'] = True
    cols = ['strata', 'codes', 'time']
    if keepStrata:
        cols += config['strata']
    df = df.loc[:, cols]

    return df


def extractCodeTimes(x, codeCols, timeCols=None):
    # Retrive unique codes and their associated time column
    codeUniq = list(np.unique(x[codeCols].dropna(), return_index=True))
    if len(codeUniq[0]) == 0:
        return pd.Series([(), ()])
    if timeCols is None:
        timeUniq = tuple([True for i in range(len(codeUniq[0]))])
    else:
        timeUniq = [timeCols[i] for i in codeUniq[1]]
        timeUniq = tuple(x[timeUniq].fillna(-1))
    # Float node names not allowed by pyvis
    if isinstance(codeUniq[0][0], float):
        codes = tuple(int(c) for c in codeUniq[0])
    else:
        codes = tuple(codeUniq[0])
    return pd.Series([codes, timeUniq])


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
