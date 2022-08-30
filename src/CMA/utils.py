#!/usr/bin/env python3

""" Helper functions for command line interface """

import re
import sys
import yaml
import json
import pprint
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
from pyvis.network import Network
from scipy.sparse import csr_matrix
import community as community_louvain
from matplotlib.colors import rgb2hex
from matplotlib.colors import Normalize
from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.proportion import proportions_ztest
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
            'input': self.mandatory,
            'edgeData': self.mandatory,
            'networkPlot': self.mandatory,
            'codes': self.mandatory,
            'refNode': [],
            'excludeNode': [],
            'radius': 1,
            'strata': None,
            'seperator': None,
            'chunkSize': None,
            'seed': 42,
            'stat': 'OR',
            'permutations': 10000,
            'minObs': 100,
            'alpha': 0.01,
            'minDegree': 0,
            'plotDPI': 300,
            'maxNodeSize': 50,
            'nodeScale': 10
        })

    def _postProcessConfig(self):
        """ Additional config modifications """
        config = self.config
        config['allCols'] = []
        if config['strata'] is not None:
            config['allCols'] += config['strata']
        # Ensure single value is in list
        for group in ['refNode', 'excludeNode']:
            if not isinstance(config[group], list):
                config[group] = [config[group]]
                # Convert to string
            config[group] = [str(node) for node in config[group]]
        refAndExclude = set(config['refNode']).intersection(config['excludeNode'])
        if refAndExclude:
            logging.error(
                f'Nodes {refAndExclude} are in reference and exclusion list')
            raise ValueError
        assert config['stat'] in ['OR', 'RR']
        intVars = ([
            'minDegree', 'radius', 'seed', 'permutations',
            'minObs', 'plotDPI', 'maxNodeSize', 'nodeScale'
        ])
        for par in intVars:
            if not isinstance(config[par], int):
                logging.error(
                    f'Non-integer argument passed to config: {par} '
                    f'({config[par]}) setting to {self.default[par]}.')
                config[par] = self.default[par]
        assert config['maxNodeSize'] > 0
        assert config['nodeScale'] > 1
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
        if config['chunkSize'] is not None:
            if (not isinstance(config['chunkSize'], int)
                    or config['chunkSize'] <= 0):
                logging.error(
                    f'Invalid chunksize {config["chunkSize"]}\n')
                self.error = True

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



def getMMFrequency(df: pd.DataFrame) -> pd.Series:
    """Process ICD-10 multi-morbidity data.

    Args:
        df (pd.DataFrame): ICD-10 data processed by CMA.processData().

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


def retrieveIndices(df: pd.DataFrame) -> list:
    """ Retrieve per-strata indices """
    indices = []
    for stratum in df['strata'].unique():
        indices.append(np.array(df['strata'] == stratum))
    return indices


def stratifiedOdds(a1, a2, indices):
    tables = makeStratifiedTable(a1, a2, indices)
    if not tables:
        return (None, None)
    minObs = np.sum(tables, axis=2).min()
    k = StratifiedTable(tables)
    return k, minObs


def proportionTest(a1, a2):
    bothTimeInfo = (a1 > 0) & (a2 > 0)
    a1_to_a2 = (a1[bothTimeInfo] < a2[bothTimeInfo]).sum()
    a2_to_a1 = (a2[bothTimeInfo] < a1[bothTimeInfo]).sum()
    z, p = proportions_ztest(
        a1_to_a2, a1_to_a2 + a2_to_a1, value=0.5, alternative='two-sided')
    # Total positive cases (ignoring time information)
    total = ((a1 != 0) & (a2 != 0)).sum()
    return z, p, total


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


def networkAnalysis(config: str, allLinks):
    allLinks = allLinks.loc[
          (~allLinks['Node1'].isin(config['excludeNode']))
        & (~allLinks['Node2'].isin(config['excludeNode']))
    ].copy()
    allNodes, allEdges = processLinks(
        allLinks, directed=config['directed'],
        stat=config['stat'], minVal=1,
        alpha=config['alpha'], minObs=config['minObs']
    )
    stat = config['stat'] # Odds ratio or Risk Ratio

    G = nx.DiGraph() if config['directed'] else nx.Graph()
    G.add_nodes_from(allNodes)
    G.add_weighted_edges_from(allEdges)

    validRefs = validateNodes(config['refNode'], G)

    G = makeEgo(G, validRefs, config['directed'], radius=config['radius'])

    if len(G.nodes()) == 1:
        logging.error(f'Reference node(s) {validRefs} have no connections.')
        return 1
    cmap = cm.viridis_r if config['refNode'] else cm.viridis
    nodeSummary = getNodeSummary(
        G, validRefs, alphaMin=0.5, size=config['maxNodeSize'],
        scale=config['nodeScale'], cmap=cmap
    )
    for node in G.nodes():
        G.nodes[node]['size'] = nodeSummary.loc[node, 'size']
        G.nodes[node]['label'] = str(node)
        G.nodes[node]['font'] = {'size': 200}
        rgb = nodeSummary.loc[node, 'colour']
        alpha = nodeSummary.loc[node, 'alpha']
        G.nodes[node]['color'] = rgb2hex((*rgb, alpha), keep_alpha=True)

    maxEdgeWidth = config['maxNodeSize'] / 20
    edgeSummary = getEdgeSummary(
        G, alphaMin=0.5, size=maxEdgeWidth, scale=config['nodeScale'])

    for edge in G.edges():
        G.edges[edge]['width'] = edgeSummary.loc[edge, 'width']
        alpha = edgeSummary.loc[edge, 'alpha']
        G.edges[edge]['color'] = rgb2hex((0, 0, 0, alpha), keep_alpha=True)

    remove = [x for x in G.nodes() if G.degree(x) < config['minDegree']]
    G.remove_nodes_from(remove)

    net = Network(height='85%', width='75%', directed=G.is_directed())
    net.from_nx(G)
    options = json.dumps({
        'edges': {'arrowStrikethrough': False},
        'physics': {
            'enabled': True,
            'barnesHut': {
                'gravitationalConstant': -80000,
                'springLength': 250,
                'springConstant': 0.001,
                'centralGravity': 0.3,
                'springStrength': 0.001,
                'damping': 0.09,
                'overlap': 0
            }
        }
    })
    net.set_options(f"var options = {options}")
    net.show(config['networkPlot'])


def makeEgo(G, refNodes, directed, radius):
    """ Subset graph based on distance to reference nodes """
    if not refNodes:
        return G
    newG = nx.DiGraph() if directed else nx.Graph()
    for ref in refNodes:
        ego = nx.ego_graph(G, n=ref, radius=radius, undirected=True)
        newG.update(ego)
    return newG.copy()


def getNodeSummary(G, refNodes, alphaMin=0.5, size=50, scale=10, cmap=cm.viridis_r):
    assert (size > 0) and (scale > 1)
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
    print(summary)
    propertiesBy = 'refDistance' if refNodes else 'Betweeness'
    summary['colour'] = setColour(summary, propertiesBy, cmap=cmap)
    reverse = True if propertiesBy == 'refDistance' else False
    # Reference ndes are the same size as the largest non-reference
    naFill = summary[propertiesBy].min()
    if (summary[propertiesBy].dropna() == naFill).all():
        summary['size'] = size
        summary['alpha'] = 1
    else:
        summary['size'] = MinMaxScaler(
            summary[propertiesBy].fillna(naFill), (size, size * scale), reverse)
        summary['alpha'] = MinMaxScaler(
            summary[propertiesBy].fillna(0), (alphaMin, 1), reverse)
    return summary


def getEdgeSummary(G, alphaMin=0.5, size=1, scale=10):
    summary = pd.DataFrame(
        {edge: G.edges[edge]['weight'] for edge in G.edges()}, index=['weight']).T
    summary['logWeight'] = np.log(summary['weight'])
    summary['alpha'] = MinMaxScaler(summary['logWeight'], (alphaMin, 1), reverse=True)
    summary['width'] = MinMaxScaler(
        summary['logWeight'], (size, size * scale), reverse=True)
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


def validateNodes(nodes, G):
    """ Check to see if all nodes are in network """
    validNodes = []
    for node in nodes:
        if node not in G.nodes():
            logging.error(f'{node} not in network.')
        else:
            validNodes.append(node)
    return validNodes


def getRefDistance(G, refNodes):
    refRGB = {}
    G = G.to_undirected() # Ignore direction for path length
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


def retrieveEdge(x, directed, stat):
    """ Return edge and correctly oriented if directed graph """
    if directed and x['zProp'] < 0:
        return (x['Node2'], x['Node1'], x[f'inverse{stat}'])
    else:
        return (x['Node1'], x['Node2'], x[f'inverse{stat}'])


def processLinks(links, directed, stat='OR', minVal=1, alpha=0.01, minObs=1):
    assert stat in links.columns
    # Force string type for pyvis
    links[['Node1', 'Node2']] = links[['Node1', 'Node2']].astype(str)
    allNodes = links[['Node1', 'Node2']].melt()['value'].drop_duplicates().tolist()
    links['ICDpair'] = [tuple(r) for r in links[['Node1', 'Node2']].to_numpy()]
    sigLinks = links.loc[
          (links['FDRnull'] < alpha)
        & (links[stat] > minVal)
        & (links['minObs'] >= minObs)
    ]
    if directed:
        sigLinks = sigLinks.loc[
              (sigLinks['FDRprop'] < alpha)
        ]
    allEdges = sigLinks.apply(
        retrieveEdge, args=(directed, stat), axis=1).tolist()
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


def reorderGroups(groups: list):
    """ Order strings numerically where possible.
    e.g. string representations of numeric intervals.
    """
    numeric = []
    str_group = [str(group) for group in groups]
    for group in str_group:
        # Remove non-numeric characters
        group = re.split(r'\D+', group)
        # Remove empty strings
        group = list(filter(None, group))
        if not group:
            # Return unchanged if no numeric
            return groups
        numeric.append(float(group[0]))
    # Sort by numeric
    groups = [groups[i] for i in np.argsort(numeric)]
    return groups
