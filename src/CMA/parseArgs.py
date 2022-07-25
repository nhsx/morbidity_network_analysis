#!/usr/bin/env python3

""" Interface for Command Line Tool. """

import sys
import CMA
import logging
import argparse
from .main import main
from .main import __doc__
from ._version import __version__


def parseArgs() -> argparse.Namespace:
    epilog = 'Stephen Richer, NHS England (stephen.richer@nhs.net)'
    baseParser = getBaseParser(__version__)
    parser = argparse.ArgumentParser(
        epilog=epilog, description=__doc__, parents=[baseParser])
    parser.set_defaults(function=main)
    return parser.parse_args()


def getBaseParser(version: str) -> argparse.Namespace:
    """ Create base parser of verbose/version. """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--version', action='version', version='%(prog)s {}'.format(version))
    parser.add_argument(
        '--verbose', action='store_const', const=logging.DEBUG,
        default=logging.ERROR, help='verbose logging for debugging')
    return parser
