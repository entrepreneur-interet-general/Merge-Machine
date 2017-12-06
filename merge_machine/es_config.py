#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:27:30 2017

@author: m75380

A list of standard Elasticsearch analyzers

"""
from .helpers import _gen_index_settings_from_analyzers

# List of dict objects defining analyzers
from .analyzers import case_insensitive_keyword, integers, n_grams
from .analyzers_resource import city

# Default analyzer (for non-matching columns)
DEFAULT_ANALYZER = 'case_insensitive_keyword'

# Default analyzers (for columns that should match)
DEFAULT_ANALYZERS = {'case_insensitive_keyword', 'city', 'integers', 'n_grams'}

# Create ES config template for custom analyzers
INDEX_SETTINGS_TEMPLATE = _gen_index_settings_from_analyzers([eval(x) for x in DEFAULT_ANALYZERS]) # TODO: change this
