#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# List of dict objects defining analyzers
from .no_resource_analyzer_definitions import special_keyword, integers, n_grams, french_estab
from .resource_analyzer_definitions import city, country

# Alias for convenience
from .gen_resources import generate_resources


# Name the analyzers to use
ANALYZERS = {'special_keyword': special_keyword,
             'integers': integers,
             'n_grams': n_grams,
             'city': city, 
             'country': country,
             'french_estab': french_estab}

# Default analyzer (for non-matching columns)
DEFAULT_ANALYZER = 'special_keyword'

# Default analyzers (for columns that should match)
CUSTOM_ANALYZERS = {'special_keyword', 'integers', 'n_grams', 'city', 'country', 'french_estab'}
STOCK_ANALYZERS = set([])
DEFAULT_ANALYZERS = CUSTOM_ANALYZERS | STOCK_ANALYZERS

