#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# List of dict objects defining analyzers
from .no_resource_analyzer_definitions import case_insensitive_keyword, integers, n_grams
from .resource_analyzer_definitions import city, country

# Alias for convenience
from .gen_resources import generate_resources


# Name the analyzers to use
ANALYZERS = {'case_insensitive_keyword': case_insensitive_keyword,
             'integers': integers,
             'n_grams': n_grams,
             'city': city, 
             'country': country}

# Default analyzer (for non-matching columns)
DEFAULT_ANALYZER = 'case_insensitive_keyword'

# Default analyzers (for columns that should match)
CUSTOM_ANALYZERS = {'case_insensitive_keyword', 'integers', 'n_grams', 'city', 'country'}
STOCK_ANALYZERS = {'french'}
DEFAULT_ANALYZERS = CUSTOM_ANALYZERS | STOCK_ANALYZERS

