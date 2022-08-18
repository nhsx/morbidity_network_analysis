#!/usr/bin/env python3

""" Generate test data for CMA. """


import sys
import argparse
import numpy as np
import pandas as pd


def simulateData(nNodes: int, nRecords: int, seed: int):
    np.random.seed(seed)
    maxMorbidity = 26
    nodes = np.array(range(1, nNodes + 1))
    allWeights = getNodeWeights(nodes)
    df = initialiseData(nodes, nRecords, maxMorbidity)
    for n in range(1, maxMorbidity):
        if n % 2 == 0:
            nodeSet = np.random.choice(nodes, nRecords)
        else:
            nodeSet = df.apply(getNextNode, args=(nodes, allWeights), axis=1)
        df[f'secondaryCode{n}'] = nodeSet
        df[f'secondaryTime{n}'] = df[f'secondaryCode{n}']
    df.drop('nMorbidities', axis=1).to_csv(sys.stdout, index=False)


def initialiseData(nodes: np.array, nRecords: int, maxMorbidity: int = 26):
    nMorbidities = np.random.geometric(1/12, nRecords)
    nMorbidities[nMorbidities > maxMorbidity] = maxMorbidity
    df = ({
        'Age': np.random.choice([5, 10, 20, 40, 80], nRecords),
        'nMorbidities': len(nodes),
        'primaryCode': np.random.choice(nodes, nRecords)
    })
    df = pd.DataFrame(df)
    df['primaryTime'] = df['primaryCode']
    return df


def getNodeWeights(nodes: np.array) -> dict:
    """ Get probability weight for each node """
    nNodes = len(nodes)
    riskRatio = nNodes
    allWeights = {}
    baseRisk = np.ones(nNodes)
    for node in nodes:
        if node == 1:
            allWeights[node] = baseRisk / baseRisk.sum()
        else:
            risk = baseRisk.copy()
            factors = getFactors(node, excludeSelf=True)
            risk[np.argwhere(np.isin(nodes, factors))] = riskRatio
            allWeights[node] = risk / risk.sum()
    return allWeights


def getFactors(n: int, excludeSelf: bool = False):
    """ Compute all factors for n. """
    factors = set() if excludeSelf else set(n)
    for i in range(1, (n // 2) + 1):
        if n % i == 0:
            factors.add(i)
    return list(factors)


def getNextNode(x, nodes, allWeights):
    allPrevious = [col for col in x.index if 'Code' in col]
    previous = allPrevious[-1]
    if len(allPrevious) >= x['nMorbidities']:
        return 'NULL'
    probs = allWeights[x[previous]].copy()
    probs[0] = (x['Age'] / 40) * probs[0]
    probs /= probs.sum()
    return np.random.choice(nodes, p=probs)
