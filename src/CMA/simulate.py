#!/usr/bin/env python3

""" Generate test data for CMA. """


import sys
import argparse
import numpy as np
import pandas as pd


def simulateData(nNodes: int, nRecords: int, weight: float, seed: int):
    assert 0 <= weight <= 1
    np.random.seed(seed)
    codesPerRecord = 6
    nodes = np.array(range(1, nNodes + 1))
    allWeights = getNodeWeights(nodes, weight)
    df = initialiseData(nodes, nRecords)
    for n in range(2, codesPerRecord + 1):
        nodeSet = df.apply(getNextNode, args=(n, nodes, allWeights), axis=1)
        df[f'code{n}'] = nodeSet
        df[f'time{n}'] = nNodes - df[f'code{n}']
    df.to_csv(sys.stdout, index=False)


def initialiseData(nodes: np.array, nRecords: int):
    df = pd.DataFrame({
        'Age': np.random.choice([10, 20, 40, 80], nRecords),
        'code1': np.random.choice(nodes, nRecords)
    })
    df['time1'] = df['code1']
    return df


def getNodeWeights(nodes: np.array, weight: int) -> dict:
    """ Get probability weight for each node """
    nNodes = len(nodes)
    allWeights = {}
    baseWeight = np.ones(nNodes)
    for node in nodes:
        if node == 1:
            allWeights[node] = baseWeight / baseWeight.sum()
        else:
            factorWeight = baseWeight.copy()
            factors = getFactors(node, excludeSelf=True)
            # Special case - zero probability of picking a factor
            if weight == 0:
                factorWeight[np.argwhere(np.isin(nodes, factors))] = 0
            # Special case - ALWAYS pick a factor
            elif weight == 1:
                factorWeight[np.argwhere(~np.isin(nodes, factors))] = 0
            else:
                allFactorWeight = weight * ((nNodes - len(factors)) / (1 - weight))
                perFactorWeight = allFactorWeight / len(factors)
                factorWeight[np.argwhere(np.isin(nodes, factors))] = perFactorWeight
            allWeights[node] = factorWeight / factorWeight.sum()
    return allWeights


def getFactors(n: int, excludeSelf: bool = False):
    """ Compute all factors for n. """
    factors = set() if excludeSelf else set(n)
    for i in range(1, (n // 2) + 1):
        if n % i == 0:
            factors.add(i)
    return list(factors)


def getNextNode(x, codeNumber, nodes, allWeights):
    previous = f'code{codeNumber-1}'
    if codeNumber % 2 != 0:
        return np.random.choice(nodes)
    else:
        probs = allWeights[x[previous]].copy()
        probs[0] = (x['Age'] / 40) * probs[0]
        probs /= probs.sum()
        return np.random.choice(nodes, p=probs)
