#!/usr/bin/python3
# coding: utf-8

import os, setuptools


readme, requirement = '', ['']
dst = 'odm_sfm'; ver = '0.1.0a12'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if os.path.isfile('README.md'):
    with open('README.md', encoding='utf-8') as f:
        readme = f.read()
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        requirement = f.read().split()

setuptools.setup(
    version = ver,
    author = 'Glodon',
    python_requires = '>=3.6',
    name = dst.replace('_','-'),
    long_description = readme,
    install_requires = requirement,
    license = 'None', # GUN LGPLv2.1
    author_email = 'fuh-d@glodon.com',
    description = 'RTK calibrate GPS.',
    package_data={'': ['*.txt','*.md']},
    url = 'https://pypi.org/project/odm-sfm/',
    packages = setuptools.find_packages(),
    #packages = setuptools.find_namespace_packages(),
    long_description_content_type = 'text/markdown',
    classifiers = ['Operating System :: POSIX :: Linux',
                    'Operating System :: Microsoft :: Windows,
                    'Programming Language :: Unix Shell',
                    'Programming Language :: Python :: 3.6+'],
    scripts = [f'{dst}/odm_sfm_bash'],
)
