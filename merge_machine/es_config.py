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

# Default analyzer
default_analyzer = 'case_insensitive_keyword'

# Create ES config template for custom analyzers
analyzers = [case_insensitive_keyword, city, integers, n_grams]
index_settings_template = _gen_index_settings_from_analyzers(analyzers)


#sample_index_settings = gen_index_settings({'nom_lycee': {'city'}})
#
#{
#    "settings" : {
#        "number_of_shards" : 1
#    },
#    "mappings" : {
#        "type1" : {
#            "properties" : {
#                "field1" : { "type" : "text" }
#            }
#        }
#    }
#}


