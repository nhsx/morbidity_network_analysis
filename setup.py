#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Template of python3 project. """

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev




import os
import sys
import glob
from shutil import rmtree
from setuptools import setup, find_namespace_packages, Command

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


def get_info():
    info = {}
    versionPath = glob.glob('src/*/_version.py')[0]
    with open(versionPath) as fp:
        exec(fp.read(), info)
    return info



setup(
    name='CMA',
    author='Stephen Richer',
    author_email='stephen.richer@nhs.net',
    url='https://github.com/StephenRicher/pyTemplateBath',
    scripts=['bin/CMA'],
    python_requires='>=3.6.0',
    install_requires=[],
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
    ],
    version=get_info()['__version__'],
    description=__doc__,
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    zip_safe=False,
    #cmdclass={
    #    'upload': UploadCommand,
    #}
)
