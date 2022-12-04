#!/usr/bin/env python3

""" Multimordibity Network Analysis """

import sys
import logging
import argparse
from timeit import default_timer as timer
from . import __version__
import morbidity_network_analysis as mma
from .main import main, edgeAnalysisOnly, networkAnalysisOnly, morbidityZ
from .simulate import simulateData


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
        description=main.__doc__,
        help='Run complete Network Analysis pipeline.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp1.add_argument(
        'config', help='YAML configuration file.')
    sp1.set_defaults(function=main)

    sp2 = subparser.add_parser(
        'process',
        description=edgeAnalysisOnly.__doc__,
        help='Pre-process data and compute edge weights.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp2.add_argument(
        'config', help='YAML configuration file.')
    sp2.set_defaults(function=edgeAnalysisOnly)

    sp3 = subparser.add_parser(
        'network',
        description=networkAnalysisOnly.__doc__,
        help='Build and visualise network.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp3.add_argument(
        'config', help='YAML configuration file.')
    sp3.set_defaults(function=networkAnalysisOnly)

    sp4 = subparser.add_parser(
        'simulate',
        description=mma.simulate.__doc__,
        help='Simulate test data.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp4.add_argument(
        '--config',
        help='Path to write default config file (default: stderr)')
    sp4.add_argument(
        '--nNodes', type=int, default=24,
        help='Total nodes in simulated network (default: %(default)s)')
    sp4.add_argument(
        '--nRecords', type=int, default=200_000,
        help='Number of records to simulate (default: %(default)s)')
    sp4.add_argument(
        '--codesPerRecord', type=int, default=4,
        help='Total codes per records (default: %(default)s)')
    sp4.add_argument(
        '--weight', type=float, default=0.9,
        help='Probability of sampling a factor (default: %(default)s)')
    sp4.add_argument(
        '--seed', type=int, default=42,
        help='Seed for random number generator (default: %(default)s)')
    sp4.set_defaults(function=simulateData)

    sp5 = subparser.add_parser(
        'strata',
        description=mma.simulate.__doc__,
        help='Estimate morbidity enrichment by strata.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp5.add_argument(
        'config', help='YAML configuration file.')
    sp5.add_argument(
        'out', help='Path to write figure.')
    sp5.add_argument(
        'morbidities', nargs='*',
        help='Morbidity set to test for enrichment.')

    sp5.set_defaults(function=morbidityZ)

    args = parser.parse_args()
    if 'function' not in args:
        parser.print_help()
        sys.exit()

    rc = executeCommand(args)
    return rc


def executeCommand(args):
    # Initialise logging
    logFormat = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    logging.basicConfig(level=args.verbose, format=logFormat)
    del args.verbose, args.command
    # Pop main function and excute script
    function = args.__dict__.pop('function')
    start = timer()
    rc = function(**vars(args))
    end = timer()
    logging.info(f'Total execution time: {end - start:.3f} seconds.')
    logging.shutdown()
    return rc


def getBaseParser(version: str) -> argparse.Namespace:
    """ Create base parser of verbose/version. """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--version', action='version', version='%(prog)s {}'.format(version))
    parser.add_argument(
        '--verbose', action='store_const', const=logging.DEBUG,
        default=logging.ERROR, help='verbose logging for debugging')
    return parser
