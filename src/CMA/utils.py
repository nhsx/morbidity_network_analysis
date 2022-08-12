#!/usr/bin/env python3

""" Helper functions for command line interface """

import sys
import yaml
import pprint
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
from pyvis.network import Network
from itertools import combinations
from scipy.sparse import csr_matrix
import community as community_louvain
from matplotlib.colors import rgb2hex
from matplotlib.colors import Normalize
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.contingency_tables import StratifiedTable


class Config():
    """ Custom class to read, validate and
        set defaults of YAML configuration file
        """

    def __init__(self, pathToYaml):
        self.error = False
        # Reserved string for mandatory arguments
        self.mandatory = 'mandatory'
        self.pathToYaml = pathToYaml
        self.config = self._readYAML()
        self._setDefault(self.config, self.default)
        self._postProcessConfig()
        if self.error:
            logging.error(
                f'Expected format:\n\n{pprint.pformat(self.default)}')
            raise ValueError('Invalid configuration.')

    def _setDefault(self, config, default, path=''):
        """ Recursively set default values. """
        for k in default:
            if isinstance(default[k], dict):
                if not isinstance(config[k], dict):
                    logging.error(
                        f'"{config[k]}" should be a dictionary.')
                    self.error = True
                else:
                    self._setDefault(
                        config.setdefault(k, {}), default[k], path=path+k)
            else:
                if (((k not in config) or (config[k] is None))
                        and (default[k] == self.mandatory)):
                    msg = f'{path}: {k}' if path else k
                    logging.error(
                        f'Missing mandatory config "{msg}".')
                    self.error = True
                config.setdefault(k, default[k])

    def _readYAML(self):
        """ Custom validation """
        with open(self.pathToYaml, 'r') as stream:
            return yaml.safe_load(stream)

    @property
    def default(self):
        """ Default values of configuration file. """
        return ({
            'file': self.mandatory,
            'codes': self.mandatory,
            'strata': None,
            'seperator': None,
            'chunksize': None,
            'refNode': [],
            'seed': 42,
            'permutations': 10000,
            'minObs': 100,
            'alpha': 0.01,
            'minDegree': 0,
            'radius': 2
        })

    def _postProcessConfig(self):
        """ Additional config modifications """
        config = self.config
        config['allCols'] = []
        if config['strata'] is not None:
            config['allCols'] += config['strata']
        # Ensure single value is in list
        if not isinstance(config['refNode'], list):
            config['refNode'] = [config['refNode']]
        # Convert to string
        config['refNode'] = [str(node) for node in config['refNode']]
        for par in ['minDegree', 'radius', 'seed', 'permutations', 'minObs']:
            if not isinstance(config[par], int):
                logging.error(
                    f'Non-integer argument passed to config: {par} '
                    f'({config[par]}) setting to {self.default[par]}.')
                config[par] = self.default[par]
        if isinstance(config['codes'], list):
            config['directed'] = False
            config['codeCols'] = config['codes']
            config['timeCols'] = None
        else:
            config['directed'] = True
            config['codeCols'] = list(config['codes'].keys())
            config['timeCols'] = list(config['codes'].values())
            config['allCols'] += config['timeCols']
        config['allCols'] += config['codeCols']
        if config['chunksize'] is not None:
            if (not isinstance(config['chunksize'], int)
                    or config['chunksize'] <= 0):
                logging.error(
                    f'Invalid chunksize {config["chunksize"]}\n')
                self.error = True

def validateCols(df, config):
    """ Check for missing columns in df """
    missingCols = set(config['allCols']) - set(df.columns)
    if missingCols:
        logging.error(f'{missingCols} not present in {config["file"]}\n')
        raise ValueError
    if config['directed']:
        timeTypes = df[config['timeCols']].select_dtypes(
            include=[np.number, np.datetime64])
        invalidType = set(config['timeCols']) - set(timeTypes.columns)
        if invalidType:
            logging.error(
                f'Invalid time type at columns {invalidType} in {config["file"]}\n')
            raise ValueError


def checkDuplicates(df, config):
    """ Check for duplicate codes in row """
    duplicates = df[config['codeCols']].apply(
        lambda x: len(set(x.dropna())) < len(x.dropna()), axis=1)

    return duplicates[duplicates].index


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


def loadData(config: dict, keepStrata: bool = False) -> pd.DataFrame:
    """ Main function for loading data """
    # Enfore codes as strings
    dtypes = {col: str for col in config['codeCols']}
    data = pd.read_csv(
        config['file'], sep=config['seperator'],
        chunksize=config['chunksize'],
        dtype=dtypes, iterator=True
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



def getMMFrequency(df: pd.DataFrame) -> pd.Series:
    """Process ICD-10 multi-morbidity data.

    Args:
        df (pd.DataFrame): ICD-10 data processed by CMA.processData().

    Returns:
         Mulimorbidity frequency of pairwise ICD-10 codes.
    """
    return (
        df['codes'].apply(
            lambda x: [tuple(sorted(x)) for x in combinations(x, 2)])
        .explode().dropna().value_counts().sort_values(ascending=False)
    )

def getICDlong(df: pd.DataFrame) -> pd.DataFrame:
    """Generate long-format ICD data, 1 code per row.

    Args:
        df (pd.DataFrame): ICD-10 data processed by CMA.processData().

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

def makeStratifiedTable(
    a1: np.array,
    a2: np.array,
    strataIndices: np.array,
    exclude: np.array = None
) -> np.array:
    """ Generate set of stratified contigency tables """
    ctTables = []
    if exclude is None:
        exclude = np.zeros(len(a1)).astype(bool)
    a1 = a1[~exclude].copy()
    a2 = a2[~exclude].copy()
    for s in strataIndices:
        ct = np.bincount(
            2 * a1[s[~exclude]].astype(bool)
            + a2[s[~exclude]].astype(bool),
            minlength=4).reshape(2,2)
        if (ct == 0).any():
            continue
        ctTables.append(ct)
    return ctTables


def retrieveIndices(df: pd.DataFrame) -> list:
    """ Retrieve per-strata indices """
    indices = []
    for stratum in df['strata'].unique():
        indices.append(np.array(df['strata'] == stratum))
    return indices


def stratifiedOdds(a1, a2, indices, directed, excludeAll=None):
    if directed:
        # Direction specific exclusion
        bothNonZero = (a1 != 0) & (a2 != 0)
        exclude = ((a1 >= a2) & bothNonZero) | excludeAll
    else:
        exclude = None
    tables = makeStratifiedTable(a1, a2, indices, exclude)
    if not tables:
        return (None, None)
    minObs = np.sum(tables, axis=2).min()
    k = StratifiedTable(tables)
    return k, minObs


def runEdgeAnalysis(df_sp, codePairs, directed):
    indices = retrieveIndices(df_sp)
    allLinks = []
    for i, ((m1, m2), count) in enumerate(codePairs.iteritems()):
        a1 = np.array(df_sp[m1])
        a2 = np.array(df_sp[m2])
        if directed:
            # Exclude amiguous / missing time stamps
            excludeAll = ((a1 == -1) & (a2 != 0)) | ((a2 == -1) & (a1 != 0))
        else:
            excludeAll = None
        k, minObs, tables = stratifiedOdds(a1, a2, indices, directed, excludeAll)
        if k is None:
            continue
        allLinks.append([
            m1, m2, count, minObs, k.oddsratio_pooled,
            k.riskratio_pooled, k.test_equal_odds().pvalue,
            k.test_null_odds().pvalue
        ])
    allLinks = pd.DataFrame(allLinks)
    allLinks.columns = (
        ['Node1', 'Node2', 'count', 'minObs', 'OR', 'RR', 'pEqual', 'pNull'])
    allLinks.loc[allLinks['pNull'].notna(), 'FDR'] = (
        fdrcorrection(allLinks.loc[allLinks['pNull'].notna(), 'pNull'])[1]
    )
    # Larger odds ratio = smaller edge
    allLinks['inverseOR'] = 1 / allLinks['OR']
    allLinks['inverseRR'] = 1 / allLinks['RR']

    return allLinks


def edgeAnalysis(config: str):
    config = Config(config).config
    df = loadData(config)
    # Retrieve full index (including records with no morbidities)
    fullIndex = getFullIndex(df)
    codePairs = getMMFrequency(df)
    df_long = getICDlong(df)
    df_sp = df2Sparse(df_long, fullIndex)
    allLinks = runEdgeAnalysis(df_sp, codePairs, config['directed'])
    return allLinks, config


def networkAnalysis(config: str, allLinks):
    allNodes, allEdges = processLinks(
        allLinks, stat='RR', minVal=1, alpha=config['alpha'], minObs=50)

    G = nx.DiGraph() if config['directed'] else nx.Graph()
    G.add_nodes_from(allNodes)
    G.add_weighted_edges_from(allEdges)
    validRefs = validateRefNode(config['refNode'], G)
    G = makeEgo(G, validRefs, config['directed'], radius=config['radius'])
    cmap = cm.viridis_r if config['refNode'] else cm.viridis
    nodeSummary = getNodeSummary(
        G, validRefs, alphaMin=0.5, size=50, scale=10, cmap=cmap)

    for node in G.nodes():
        G.nodes[node]['size'] = nodeSummary.loc[node, 'size']
        G.nodes[node]['label'] = str(node)
        G.nodes[node]['font'] = {'size': 200}
        rgb = nodeSummary.loc[node, 'colour']
        alpha = nodeSummary.loc[node, 'alpha']
        G.nodes[node]['color'] = rgb2hex((*rgb, alpha), keep_alpha=True)

    allEdges = {edge: 1 / G.edges[edge]['weight'] for edge in G.edges()}
    allEdges = pd.Series(allEdges).to_frame().rename({0: 'OR'}, axis=1)
    allEdges['logOR'] = np.log(allEdges['OR'])
    allEdges['scaled'] = MinMaxScaler(allEdges['logOR'], (0.1, 1))
    # Truncate to 1 in case of rounding error
    allEdges['scaled'] = allEdges['scaled'].apply(lambda x: x if x < 1 else 1)
    allEdges = allEdges['scaled'].to_dict()

    for edge in G.edges():
        # Invert weight to get larger width for larger odds ratio
        G.edges[edge]['width'] = np.log(1 / G.edges[edge]['weight'])
        G.edges[edge]['color'] = rgb2hex((0, 0, 0, allEdges[edge]), keep_alpha=True)

    remove = [x for x in G.nodes() if G.degree(x) < config['minDegree']]
    G.remove_nodes_from(remove)

    net = Network(height='85%', width='75%', directed=G.is_directed())
    net.from_nx(G)
    net.toggle_physics(True)
    net.barnes_hut()
    net.show('exampleNet.html')


def makeEgo(G, refNodes, directed, radius):
    """ Subset graph based on distance to reference nodes """
    if not refNodes: return G
    newG = nx.DiGraph() if directed else nx.Graph()
    for ref in refNodes:
        newG.update(nx.ego_graph(G, n=ref, radius=radius, undirected=True))
    return newG.copy()


def getNodeSummary(G, refNodes, alphaMin=0.5, size=50, scale=10, cmap=cm.viridis_r):
    centrality = pd.Series(nx.betweenness_centrality(G,  weight='weight'), name='Betweeness')
    degree = pd.DataFrame(G.degree()).set_index(0)[1].rename('Degree')
    summary = pd.concat([centrality, degree], axis=1)
    if G.is_directed():
        logging.warning('Community detection not supported for directed graphs.')
    else:
        partitionRGB = getNodePartion(G)
        summary = pd.concat([summary, partitionRGB], axis=1)
    if refNodes:
        refRGB = getRefDistance(G, refNodes)
        summary = pd.concat([summary, refRGB], axis=1)
    size=50
    scale=10
    assert (size > 0) and (scale > 1)
    propertiesBy = 'refDistance' if refNodes else 'Betweeness'

    summary['colour'] = setColour(summary, propertiesBy, cmap=cmap)
    reverse = True if propertiesBy == 'refDistance' else False
    summary['size'] = MinMaxScaler(
        summary[propertiesBy].fillna(0), (size, size * scale), reverse)
    summary['alpha'] = MinMaxScaler(
        summary[propertiesBy].fillna(0), (alphaMin, 1), reverse)
    return summary


def setColour(summary, colourBy, cmap=cm.viridis_r):
    values = summary[colourBy].dropna()
    norm = Normalize(
        vmin=np.nanmin(summary[colourBy]), vmax=np.nanmax(summary[colourBy]))
    return summary[colourBy].apply(
        lambda x: (0,0,0) if np.isnan(x) else cmap(norm(x))[:3]
    )


def getNodePartion(G, colours=None):
    if colours is None:
        colours = ([
            (34,136,51), (204,187,68), (238,102,119),
            (170,51,119), (68,119,170), (102,204,238),
            (187,187,187)
        ])
    otherColour = colours[-1]
    partitionColours = colours[:-1]
    allPartitions = community_louvain.best_partition(G)
    # Get largest partitions in network
    mainPartions = (
        pd.Series(allPartitions.values())
        .value_counts()
        .head(len(partitionColours)).index
    )
    mainPartions = dict(zip(mainPartions, partitionColours))
    partitionInfo = {}
    for node, partition in allPartitions.items():
        if partition not in mainPartions:
            partitionInfo[node] = (otherColour, partition)
        else:
            partitionInfo[node] = (mainPartions[partition], partition)
    partitionInfo = (
        pd.DataFrame(partitionInfo).T
        .rename({0: 'partitionRGB', 1: 'partition'}, axis=1)
    )
    # Convert RGB to [0, 1] scale
    partitionInfo['partitionRGB'] = partitionInfo['partitionRGB'].apply(
        lambda x: (x[0] / 255, x[1] / 255, x[2] / 255)
    )
    return partitionInfo


def validateRefNode(refNodes, G):
    """ Check to see if all reference nodes are in network """
    validRefs = []
    for ref in refNodes:
        if ref not in G.nodes():
            logging.error(f'{ref} not in network.')
        else:
            validRefs.append(ref)
    return validRefs


def getRefDistance(G, refNodes):
    refRGB = {}
    for node in sorted(G.nodes()):
        if node in refNodes:
            refRGB[node] = np.nan
            continue
        for i, ref in enumerate(refNodes):
            if nx.has_path(G, ref, node):
                dist = nx.dijkstra_path_length(G, ref, node, weight='weight')
            else:
                dist = np.nan
            # Set value for first check
            if (i == 0):
                refRGB[node] = dist
            elif not np.isnan(dist):
                refRGB[node] = np.nanmin([refRGB[node], dist])
    return pd.Series(refRGB).to_frame().rename({0: 'refDistance'}, axis=1)



def processLinks(links, stat='OR', minVal=1, alpha=0.01, minObs=1):
    assert stat in links.columns
    # Force string type for pyvis
    links[['Node1', 'Node2']] = links[['Node1', 'Node2']].astype(str)
    allNodes = links[['Node1', 'Node2']].melt()['value'].drop_duplicates().tolist()
    links['ICDpair'] = [tuple(r) for r in links[['Node1', 'Node2']].to_numpy()]
    sigLinks = links.loc[
        (links['FDR'] < alpha)
        & (links[stat] > minVal)
        & (links['minObs'] >= minObs)
    ]
    allEdges = sigLinks.apply(
        lambda x: (x['Node1'], x['Node2'], x[f'inverse{stat}']), axis=1).tolist()
    return allNodes, allEdges


def MinMaxScaler(x, feature_range=(0, 1), reverse=False):
    """ Custom MinMax function that allows for reverse scaling """
    x = np.array(x)
    minV, maxV = feature_range
    if reverse:
        X_std = (max(x) - x) /  (max(x) - min(x))
    else:
        X_std = (x - min(x)) / (max(x) - min(x))
    X_scaled = X_std * (maxV - minV) + minV
    return X_scaled


def makeChunks(nReps, chunkSize):
    if nReps <= chunkSize:
        return [nReps]
    chunks = [chunkSize for i in range((nReps // chunkSize))]
    remain = nReps % chunkSize
    if remain > 0:
        chunks.append(remain)
    return chunks


def stratifiedPermute(df, stratifyBy, ref, n):
    """ Stratified permutation of ref values """
    originalIndex = df.index
    df = df.set_index(stratifyBy)[[ref]]
    return (
        pd.DataFrame(np.tile(df.values, n), index=df.index)
        .groupby(df.index)
        .transform(np.random.permutation)
        .set_index(originalIndex)
    )


def permutationTest(df, stratifyBy, group, ref, nReps, chunkSize=10000):
    null = []
    allData = []
    for chunk in makeChunks(nReps, chunkSize):
        p = stratifiedPermute(df, stratifyBy, ref, chunk)
        p[group] = df[group]
        null.append(p.groupby(group).sum())
    null = pd.concat(null, axis=1)

    agg = null.agg(['mean', 'std'], axis=1)
    agg[['statistic', 'count']] = df.groupby(group)[ref].agg(['sum', 'size'])
    agg['z'] = ((agg['statistic'] - agg['mean']) / agg['std'])

    return agg
