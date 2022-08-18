#!/usr/bin/env python3

from .utils import *
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(config: str):
    """ Run Full Network Analysis Pipeline """
    allLinks, config = edgeAnalysis(config)
    networkAnalysis(config, allLinks)


def edgeAnalysisOnly(config: str):
    """ Run Stage 1 of Network Analysis Pipeline """
    allLinks, config = edgeAnalysis(config)
    allLinks.to_csv(config['edgeData'], index=False)


def networkAnalysisOnly(config: str):
    """ Run Stage 2 of Network Analysis Pipeline """
    config = Config(config).config
    dtype = ({
        'Node1': str, 'Node2': str, 'count': str,
        'minObs': int, 'OR': float, 'RR': float,
        'pEqual': float, 'pNull': float, 'FDR': float,
        'inverseOR': float, 'inverseRR': float
    })
    allLinks = pd.read_csv(config['edgeData'], dtype=dtype)
    networkAnalysis(config, allLinks)


def morbidityZ(config: str, out: str, morbidities: list):
    config = Config(config).config
    np.random.seed(config['seed'])

    df = loadData(config, keepStrata=True)
    strata = config['strata']
    if len(strata) == 1:
        assert 'tempCol' not in df.columns
        df['tempCol'] = True
        plotgrid = (1,)
    else:
        plotgrid = (round(len(strata) / 2), 2)

    morbidities = set(tuple(morbidities))
    df['pair'] = df['codes'].apply(lambda x: morbidities.issubset(x))
    fig, axes = plt.subplots(*plotgrid, figsize=(16,9))
    axes = [axes] if len(strata) == 1 else axes.flatten()

    for i, stratum in enumerate(strata):
        if len(strata) == 1:
            stratifyBy = ['tempCol']
        else:
            stratifyBy = [x for x in strata if x != stratum]
        agg = permutationTest(
            df, stratifyBy, stratum, ref='pair',
            nReps=config['permutations'], chunkSize=2000)
        # Exclude groups with too few positive samples
        agg.loc[agg['statistic'] < config['minObs'], 'z'] = np.nan
        axes[i].axhline(0, color='black', ls='--')
        axes[i].axhline(2.576, color='grey', ls='--')
        axes[i].axhline(-2.576, color='grey', ls='--')
        sns.barplot(
            x=stratum, y='z', data=agg.reset_index(),
            order=reorderGroups(list(agg.index)), ax=axes[i]
        )
        if i % 2 == 0:
            axes[i].set_ylabel('Z (std from exp. mean)')
        else:
            axes[i].set_ylabel('')
        axes[i].set_xlabel(stratum)

    # Turn off blank axis
    if len(axes) > len(strata):
        axes[i+1].axis('off')

    fig.suptitle(f'{morbidities}')
    fig.tight_layout()
    fig.savefig(out, dpi=config['plotDPI'])
