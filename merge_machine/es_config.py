#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:27:30 2017

@author: m75380

A list of standard Elasticsearch analyzers

"""
import copy
import os



def _gen_index_settings_template(analyzers):
    '''
    Takes our own custom analyzer definitions and turns them into appropriate
    input for Elasticsearch settings.
    '''
    index_settings_template = {
        "settings": {
            "analysis": {
                "tokenizer": {},
                "filter": {},
                "analyzer": {}
            }
        }
    }
    
    for analyzer in analyzers:
        for key in ['tokenizer', 'filter', 'analyzer']:
            # TODO: check that keys are not overwritten
            index_settings_template['settings']['analysis'][key].update(analyzer.get(key, {}))
    return index_settings_template

 
def _gen_index_settings(default_analyzer, columns_to_index, index_settings=None):
    '''
    INPUT:
        - default_analyzer: the analyzer that will be used on all fields mentioned
            in columns_to_index no matter what
        - columns_to_index: the analyzers to use for each column
        - settings: (optional) Elasticsearch settings if to define custom 
                    analyzers if any are used. See Elasticsearch documentation 
                    on how to create custom analyzers
    '''
    # 
    index_settings = copy.deepcopy(index_settings_template)
    
    # Define mappings
    field_mappings = {
        key: {
            'analyzer': default_analyzer,
            'type': 'string',
            'fields': {
                analyzer: {
                    'type': 'string',
                    'analyzer': analyzer
                }
                for analyzer in values
            }
        }
        for key, values in columns_to_index.items() if values
    }
                
    field_mappings.update({
        key: {
            'analyzer': default_analyzer,
            'type': 'string'
        }
        for key, values in columns_to_index.items() if not values
    })
                
    assert 'mappings' not in index_settings
    index_settings['mappings'] = {'structure': {'properties': field_mappings}}
    
    return index_settings    
    
    
    
def _gen_index_settings_old(index_settings_template, columns_to_index, default_analyzer):
    '''
    Creates the dict to pass to index creation based on the columns_to_index
    
    NB: the default analyzer is keyword
    
    INPUT:
        - index_settings_template
        - columns_to_index
        
            Ex:
                {
                "nom_lycee": {
                        'french', 'whitespace', 'integers', 'n_grams', 'city'
                        }, 
                "ville": {
                        'french', 'whitespace', 'city'
                        },
                "ID": {}
                }
                
    #TODO: test if some analyzers exist
    '''
    index_settings = copy.deepcopy(index_settings_template)
    
    field_mappings = {
        key: {
            'analyzer': default_analyzer,
            'type': 'string',
            'fields': {
                analyzer: {
                    'type': 'string',
                    'analyzer': analyzer
                }
                for analyzer in values
            }
        }
        for key, values in columns_to_index.items() if values
    }
                
    field_mappings.update({
        key: {
            'analyzer': default_analyzer,
            'type': 'string'
        }
        for key, values in columns_to_index.items() if not values
    })
                
    index_settings['mappings']['structure']['properties'] = field_mappings
    
    return index_settings


# List of dict objects defining analyzers
from .analyzers import case_insensitive_keyword, integers, n_grams
from .analyzers_resource import city

analyzers = [case_insensitive_keyword, city, integers, n_grams]
index_settings_template = _gen_index_settings_template(analyzers)

# Create
gen_index_settings = lambda default_analyzer, columns_to_index: \
   _gen_index_settings(default_analyzer, columns_to_index, index_settings_template)

#
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


