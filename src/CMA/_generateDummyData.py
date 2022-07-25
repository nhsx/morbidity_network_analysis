#!/usr/bin/env python3

""" Generate dummy test data for CMA. """


import sys
import argparse
import collections
import numpy as np


def generateData(
        nodes: int, seed: int, groupSize: int,
        nPatients: int, weight: int, overlap: int):
    np.random.seed(seed)
    # Probability of having N morbidities
    multimordibity = ({
          0: 0.05,  1: 0.05,  2: 0.05, 5: 0.75, 10: 0.10
    })

    # Define CSV header
    strataHead = ['sex', 'age']
    primaryHead = ['Primary_Diagnosis_Code']
    codeHeaders = (
          ['Primary_Diagnosis_Code']
        + [f'Secondary_Diagnosis_Code_{i:02d}' for i in range(1, 25 + 1)]
    )
    timeHeaders = (
           ['Primary_Diagnosis_Time']
        + [f'Secondary_Diagnosis_Time_{i:02d}' for i in range(1, 25 + 1)]
    )
    header = strataHead + codeHeaders + timeHeaders

    print(*header, sep=',')
    for i in range(nPatients):
        sex = np.random.choice(['male', 'female'])
        age = np.random.choice([25, 55])
        # Select number of morbidities
        nMorbidities = selectedWeightValue(multimordibity)
        nEmpty = 26 - nMorbidities
        if nMorbidities == 0:
            simulated = [sex, age] + ((nEmpty * 2) * ['NULL'])
        else:
            simulatedMM = sampleNodes(nodes, nMorbidities, overlap, weight)
            # Select time as the node value (enforce directionality)
            simTime = simulatedMM.copy() + 1
            # Shuffle time order sometimes to add some noise
            #if np.random.random() < 0.25:
            #    np.random.shuffle(simTime)
            simulatedMM = np.concatenate([simulatedMM, nEmpty * ['NULL']])
            simTime = np.concatenate([simTime, nEmpty * ['NULL']])
            # Write output
            simulated = np.concatenate([[sex, age], simulatedMM, simTime])
        print(*simulated, sep=',')


def selectedWeightValue(d: dict) -> int:
    """ Function to randomly selected a value from
        dictionary of keys and weights"""
    choices = list(d.keys())
    weights = np.array(list(d.values()))
    weights /= weights.sum()
    return np.random.choice(choices, p=weights)


def sampleNodes(nodes: int, size: int, overlap: int, weight: float):
    baseP = np.ones(nodes)
    #baseP[::10] = baseP[::10] * weight
    # Intialise selection
    select = np.random.choice(range(nodes))
    allMM = [select]
    for i in range(size - 1):
        previousNode = allMM[i]
        # Lonely nodes - no dependence for high value nodes
        if previousNode > 50:
            select = np.random.choice(
                range(nodes), p=(np.ones(nodes)) / nodes)
        else:
            a = ((previousNode // 10) * 10) - overlap
            b = (a + 10) + overlap
            # Prevent index error
            a = max(0, a)
            b = min(50, min(len(baseP) - 1, b))
            p = baseP.copy()
            if previousNode % 2 == 0:
                p[a:b:2] = weight
            else:
                p[a+1:b:2] = weight
            p /= p.sum()
            while True:
                select = np.random.choice(list(range(nodes)), p=p)
                if select not in allMM:
                    break
        allMM.append(select)
    return np.array(allMM)


def parseArgs():

    epilog = 'Stephen Richer, NHS England (stephen.richer@nhs.net)'
    parser = argparse.ArgumentParser(epilog=epilog, description=__doc__)
    parser.add_argument(
        '--nodes', type=int, default=60,
        help='Total nodes in simulated network (default: %(default)s)')
    parser.add_argument(
        '--nPatients', type=int, default=20_000,
        help='Number of patient records to simulate (default: %(default)s)')
    parser.add_argument(
        '--groupSize', type=int, default=10,
        help='Group size of co-occuring clusters (default: %(default)s)')
    parser.add_argument(
        '--weight', type=float, default=5,
        help='Sampling weight factor for associated groups (default: %(default)s)')
    parser.add_argument(
        '--overlap', type=int, default=1,
        help='Co-occurence overlap (default: %(default)s)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Seed for random number generator (default: %(default)s)')
    parser.set_defaults(function=generateData)
    args = parser.parse_args()
    function = args.function
    del args.function

    return args, function


if __name__ == '__main__':
    args, function = parseArgs()
    sys.exit(function(**vars(args)))
