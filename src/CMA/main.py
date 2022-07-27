#!/usr/bin/env python3

from .utils import *
import sys
import pandas as pd


def main(config: str):
    """ Run Full Network Analysis Pipeline """
    allLinks, config = edgeAnalysis(config)
    networkAnalysis(config, allLinks)


def edgeAnalysisOnly(config: str, out: str = sys.stdout):
    """ Run Stage 1 of Network Analysis Pipeline """
    allLinks, config = edgeAnalysis(config)
    allLinks.to_csv(out)


def networkAnalysisOnly(config: str, edgeData: str):
    """ Run Stage 2 of Network Analysis Pipeline """
    config = Config(config).config
    allLinks = pd.read_csv(edgeData)
    networkAnalysis(config, allLinks)
