#!/usr/bin/python3
# coding: utf-8

import os, setuptools

readme, requirement = '', ['']
dst = 'odm_sfm'; ver = '0.1.0a7'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if os.path.isfile('README.md'):
    with open('README.md', encoding='utf-8') as f:
        readme = f.read()
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        requirement = f.read().split()
# if os.path.isfile('src/check_gz.cpp'):
#     os.system(f'g++ src/check_gz.cpp -o {dst}/check')


setuptools.setup(
    version = ver,
    author = 'Glodon',
    platforms = 'Ubuntu',
    python_requires = '>=3.6',
    name = dst.replace('_','-'),
    license = 'None', # GUN LGPLv2.1
    author_email = 'fuh-d@glodon.com',
    description = 'RTK calibrate GPS.',
    url = 'https://pypi.org/project/odm-sfm/',
    long_description = readme,
    install_requires = requirement,
    long_description_content_type = 'text/markdown',
    package_data={'': ['check','*.txt','*.md']},
    packages = setuptools.find_packages(),
    #packages = setuptools.find_namespace_packages(),
    classifiers = ['Operating System :: POSIX :: Linux',
                    'Programming Language :: Unix Shell',
                    'Programming Language :: Python :: 3.6'],
    # entry_points = {'console_scripts': [
    #                 f'odm_lla = {dst}.ODM_SfM:ODM_img_lla2',
    #                 f'odm_log = {dst}.ODM_SfM:parse_log'] },
    scripts = [f'{dst}/odm_sfm_lla'],
)
