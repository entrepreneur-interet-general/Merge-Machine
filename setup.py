#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:16:56 2017

@author: m75380
"""

from distutils.core import setup

setup(
  name = 'merge_machine',
  packages = ['merge_machine'], # this must be the same as the name above
  version = '0.1.2',
  description = 'A library for extreme fuzzy tabular data matching that relies on Elasticsearch',
  author = 'EIG 2017',
  author_email = 'leo.bou@gmail.com',
  url = 'https://github.com/eig-2017/Merge-Machine', # use the URL to the github repo
  download_url = 'https://github.com/eig-2017/Merge-Machine/archive/0.1.2.tar.gz',
  keywords = ['csv', 'link', 'merge', 'elasticsearch'],
  classifiers = [],
  python_requires='>=3',
)
