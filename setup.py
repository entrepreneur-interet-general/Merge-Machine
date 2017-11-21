#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:16:56 2017

@author: m75380
"""

from distutils.core import setup

from sys import version_info

class NotSupportedException(BaseException): pass

if version_info.major < 3:
    raise NotSupportedException("Only Python 3.x Supported")  


setup(
  name = 'merge_machine',
  packages = ['merge_machine'], # this must be the same as the name above
  version = '0.1',
  description = 'A library for extreme fuzzy tabular data matching that relies on Elasticsearch',
  author = 'EIG 2017',
  author_email = 'leo.bou@gmail.com',
  # url = 'https://github.com/XXXXXXX', # use the URL to the github repo
  # download_url = 'https://github.com/peterldowns/mypackage/archive/0.1.tar.gz', # I'll explain this in a second
  # keywords = ['testing', 'logging', 'example'], # arbitrary keywords
  classifiers = [],
)
