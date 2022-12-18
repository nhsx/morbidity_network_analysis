#!/usr/bin/env python3

""" MultiNet - Multimordibity Network Analysis """

import sys
import logging
import argparse
from timeit import default_timer as timer
from . import __version__
import multinet
from .main import edge_analysis_cli, network_analysis_cli, enrichment_analysis_cli
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

    sp2 = subparser.add_parser(
        'process',
        description=edge_analysis_cli.__doc__,
        help='Pre-process data and compute edge weights.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp2.add_argument(
        'config', help='YAML configuration file.')
    sp2.set_defaults(function=edge_analysis_cli)

    sp3 = subparser.add_parser(
        'network',
        description=network_analysis_cli.__doc__,
        help='Build and visualise network.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp3.add_argument('config', help='YAML configuration file.')
    sp3.add_argument(
        '--display', action='store_true',
        help='Display visuals after creation (default: %(default)s)')
    sp3.set_defaults(function=network_analysis_cli)

    sp4 = subparser.add_parser(
        'simulate',
        description=multinet.simulate.__doc__,
        help='Simulate test data.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp4.add_argument('out', help='Path to write output file.')
    sp4.add_argument(
        '--config',
        help='Path to write default config file (default: stderr)')
    sp4.add_argument(
        '--nNodes', type=int, default=30,
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
        'enriched',
        description=enrichment_analysis_cli.__doc__,
        help='Estimate morbidity enrichment by strata.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp5.add_argument(
        'config', help='YAML configuration file.')
    sp5.add_argument(
        '--display', action='store_true',
        help='Display visuals after creation (default: %(default)s)')
    sp5.set_defaults(function=enrichment_analysis_cli)

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
