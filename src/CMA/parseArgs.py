#!/usr/bin/env python3

""" Interface for Command Line Tool. """


import sys
import logging
import argparse
from .main import main
from ._version import __version__


def parseArgs():

    epilog = 'Stephen Richer, NHS England (stephen.richer@nhs.net)'
    parser = argparse.ArgumentParser(epilog=epilog, description=__doc__)
    parser.add_argument(
        '--version', action='version',
        version='%(prog)s {}'.format(__version__))
    parser.add_argument(
        '--verbose', action='store_const', const=logging.DEBUG,
        default=logging.ERROR, help='verbose logging for debugging')
    parser.set_defaults(function=main)

    args = parser.parse_args()
    logFormat = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    logging.basicConfig(level=args.verbose, format=logFormat)
    function = args.function
    del args.verbose, args.function

    return args, function
