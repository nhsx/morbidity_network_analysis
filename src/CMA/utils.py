#!/usr/bin/env python3

""" Helper functions for command line interface """

import yaml
import pprint
import logging


class Config():
    """ Custom class to read, validate and
        set defaults of YAML configuration file
        """

    def __init__(self, pathToYaml):
        self.error = False
        # Reserved string for mandatory arguments
        self.mandatory = 'mandatory'
        self.pathToYaml = pathToYaml
        self.config = self._readYAML()
        self._setDefault(self.config, self.default)
        self._postProcessConfig()
        if self.error:
            logging.error(
                f'Expected format:\n\n{pprint.pformat(self.default)}')
            raise ValueError('Invalid configuration.')

    def _setDefault(self, config, default, path=''):
        """ Recursively set default values. """
        for k in default:
            if isinstance(default[k], dict):
                if not isinstance(config[k], dict):
                    logging.error(
                        f'"{config[k]}" should be a dictionary.')
                    self.error = True
                else:
                    self._setDefault(
                        config.setdefault(k, {}), default[k], path=path+k)
            else:
                if (((k not in config) or (config[k] is None))
                        and (default[k] == self.mandatory)):
                    msg = f'{path}: {k}' if path else k
                    logging.error(
                        f'Missing mandatory config "{msg}".')
                    self.error = True
                config.setdefault(k, default[k])

    def _readYAML(self):
        """ Custom validation """
        with open(self.pathToYaml, 'r') as stream:
            return yaml.safe_load(stream)

    @property
    def default(self):
        """ Default values of configuration file. """
        return ({
            'file': self.mandatory,
            'codes': self.mandatory,
            'strata': None,
            'seperator': None,
            'chunksize': None,
        })

    def _postProcessConfig(self):
        """ Additional config modifications """
        config = self.config
        config['allCols'] = []
        if config['strata'] is not None:
            config['allCols'] += config['strata']
        if isinstance(config['codes'], list):
            config['directed'] = False
            config['codeCols'] = config['codes']
            config['timeCols'] = None
        else:
            config['directed'] = True
            config['codeCols'] = list(config['codes'].keys())
            config['timeCols'] = list(config['codes'].values())
            config['allCols'] += config['timeCols']
        config['allCols'] += config['codeCols']
        if config['chunksize'] is not None:
            if (not isinstance(config['chunksize'], int)
                    or config['chunksize'] <= 0):
                logging.error(
                    f'Invalid chunksize {config["chunksize"]}\n')
                self.error = True
