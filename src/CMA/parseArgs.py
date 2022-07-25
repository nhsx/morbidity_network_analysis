#!/usr/bin/env python3

""" Multimordibity Network Analysis """

import sys
import CMA
import logging
import argparse
from .main import main
from .simulate import simulateData
from ._version import __version__


def parseArgs() -> argparse.Namespace:
    epilog = 'Stephen Richer, NHS England (stephen.richer@nhs.net)'
    baseParser = getBaseParser(__version__)
    parser = argparse.ArgumentParser(
        epilog=epilog, description=__doc__, parents=[baseParser])
    subparser = parser.add_subparsers(
        title='required commands',
        description='',
        dest='command',
        metavar='Commands',
        help='Description:')

    sp1 = subparser.add_parser(
        'analyse',
        description=CMA.main.__doc__,
        help='Run Network Analysis.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp1.add_argument(
        'config',
        help='Provide name.')
    sp1.set_defaults(function=main)

    sp2 = subparser.add_parser(
        'simulate',
        description=CMA.simulate.__doc__,
        help='Simulate morbidity data.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp2.add_argument(
        '--nodes', type=int, default=60,
        help='Total nodes in simulated network (default: %(default)s)')
    sp2.add_argument(
        '--nRecords', type=int, default=20_000,
        help='Number of patient records to simulate (default: %(default)s)')
    sp2.add_argument(
        '--weight', type=float, default=5,
        help='Sampling weight for associated groups (default: %(default)s)')
    sp2.add_argument(
        '--overlap', type=int, default=1,
        help='Co-occurence overlap (default: %(default)s)')
    sp2.add_argument(
        '--seed', type=int, default=42,
        help='Seed for random number generator (default: %(default)s)')
    sp2.set_defaults(function=simulateData)

    args = parser.parse_args()
    if 'function' not in args:
        parser.print_help()
        sys.exit()

    return args


def getBaseParser(version: str) -> argparse.Namespace:
    """ Create base parser of verbose/version. """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--version', action='version', version='%(prog)s {}'.format(version))
    parser.add_argument(
        '--verbose', action='store_const', const=logging.DEBUG,
        default=logging.ERROR, help='verbose logging for debugging')
    return parser
