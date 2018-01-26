#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:27:30 2017

@author: m75380

List global variables defining standard analyzers to use and their definitions.
"""
from .helpers import _gen_index_settings_from_analyzers

# List of dict objects defining analyzers
from .analyzers import case_insensitive_keyword, integers, n_grams
from .analyzers_resource import city, country

# Default analyzer (for non-matching columns)
DEFAULT_ANALYZER = 'case_insensitive_keyword'

# Default analyzers (for columns that should match)
DEFAULT_CUSTOM_ANALYZERS = {'case_insensitive_keyword', 'integers', 'n_grams'}
DEFAULT_STOCK_ANALYZERS = {'french'}
DEFAULT_ANALYZERS = DEFAULT_CUSTOM_ANALYZERS | DEFAULT_STOCK_ANALYZERS

# Create ES config template for custom analyzers
INDEX_SETTINGS_TEMPLATE = _gen_index_settings_from_analyzers([eval(x) for x in DEFAULT_CUSTOM_ANALYZERS]) # TODO: change this
