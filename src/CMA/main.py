#!/usr/bin/env python3

""" Perform Multi-Morbidity Network Analysis. """

from .utils import *


def main(config: str):
    config = Config(config)
    print(config.config)
