#!/usr/bin/env python3

from .utils import *
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multinet.edgeAnalysis as ea


def main(config: str):
    """ Run Full Network Analysis Pipeline """
    allLinks, config = ea.edgeAnalysis(config)
    allLinks.to_csv(config['edgeData'], index=False)
    networkAnalysis(config, allLinks)


def edgeAnalysisOnly(config: str):
    """ Run Stage 1 of Network Analysis Pipeline """
    allLinks, config = ea.edgeAnalysis(config)
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


def morbidityZ(config: str):
    config = Config(config).config
    np.random.seed(config['seed'])
    sns.set(font_scale=2, style='white')
    if not config['enrichmentNode']:
        logging.error('No node(s) provided to config["enrichmentNode"].')
        return 1
    elif config['demographics'] is None:
        logging.error('No strata provided to config["demographics"].')
        return 1
    elif config['enrichmentPlot'] is None:
        logging.error('No output filename provided to config["enrichmentPlot"].')
        return 1
    df = loadData(config, keepCols=config['demographics'])
    demo = config['demographics']
    if len(demo) == 1:
        assert 'tempCol' not in df.columns
        df['tempCol'] = True
        plotgrid = (1,)
    else:
        plotgrid = (round(len(demo) / 2), 2)

    morbidities = set(tuple(config['enrichmentNode']))
    df['pair'] = df['codes'].apply(lambda x: not morbidities.isdisjoint(x))
    fig, axes = plt.subplots(*plotgrid, figsize=(16,9))
    axes = [axes] if len(demo) == 1 else axes.flatten()

    for i, stratum in enumerate(demo):
        if len(demo) == 1:
            stratifyBy = ['tempCol']
        else:
            stratifyBy = [x for x in demo if x != stratum]

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
    if len(axes) > len(demo):
        axes[i+1].axis('off')

    fig.suptitle(f'{morbidities}')
    fig.tight_layout()
    fig.savefig(config['enrichmentPlot'])
