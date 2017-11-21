#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:27:30 2017

@author: m75380
"""
import copy
import os

#curdir = os.path.dirname(os.path.realpath(__file__))
#os.chdir(curdir)

#city_keep_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_keep.txt')
#city_syn_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_synonyms.txt')

organization_keep_file_path = 'es_organization_keep.txt'
organization_syn_file_path = 'es_organization_synonyms.txt'

city_keep_file_path = 'es_city_keep.txt'
city_syn_file_path = 'es_city_synonyms.txt'


tokenizers = {
    "integers": {
        "type": "pattern",
        "preserve_original": 0,
        "pattern": '(\\d+)',
        'group': 1
    },
    "char_n_grams": {
        "type": "ngram",
        "min_gram": 3,
        "max_gram": 3,
        "token_chars": [
            "letter",
            "digit"
        ]
    },
    "my_standard": { #standard-ish except for "-"
        "type": "pattern",
        "pattern": '|'.join(["\'", '\"', '\(', '\)', '_', ',', '\.', ';'])
    }
}

filters = {
    "my_edgeNGram": {
        "type": "edgeNGram",
        "min_gram": 3,
        "max_gram": 30
    },
    "my_length": {
        "type" : "length",
        "min": 4
    },
            
            
    "leading_zero_trim": {
        "type": "pattern_replace",
        "pattern": "^0+(.*)",
        "replacement": "$1"
    },

# =============================================================================
# French re-implement
# =============================================================================
#    "french_elision": {
#      "type":         "elision",
#      "articles_case": True,
#      "articles": [
#          "l", "m", "t", "qu", "n", "s",
#          "j", "d", "c", "jusqu", "quoiqu",
#          "lorsqu", "puisqu"
#        ]
#    },
#    "french_stop": {
#      "type":       "stop",
#      "stopwords":  "_french_" 
#    },
#    "french_keywords": {
#      "type":       "keyword_marker",
#      "keywords":   ["Exemple"] 
#    },
#    "french_stemmer": {
#      "type":       "stemmer",
#      "language":   "light_french"
#    },
                
# =============================================================================
#   Organization filters    
# =============================================================================
#    "my_org_keep":{
#        "type" : "keep",
#        "keep_words_case": True,
#        "keep_words_path": organization_keep_file_path      
#    },
#
#    "my_org_stop":{
#        "type" : "stop",
#        "ignore_case": True,
#        "stopwords_path": organization_keep_file_path      
#    },
#
#    "my_org_synonym" : {
#        "type" : "synonym", 
#        "expand": False,    
#        "ignore_case": True,
#        "synonyms_path" : organization_syn_file_path,
#        "tokenizer" : "my_standard"  # TODO: whitespace? 
#    },   

# =============================================================================
# City filters
# =============================================================================
    "my_city_keep" : {
        "type" : "keep",
        "keep_words_case": True, # Lower the words
        "keep_words_path" : city_keep_file_path
    },
            
    "my_city_stop":{
        "type" : "stop",
        "ignore_case": True,
        "stopwords_path": city_keep_file_path      
    },
      
    "my_city_synonym" : {
        "type" : "synonym", 
        "expand": False,    
        "ignore_case": True,
        # "synonyms" : ["paris, lutece => paname"],
        "synonyms_path" : city_syn_file_path,
        "tokenizer" : "my_standard"  # TODO: whitespace? 
    }
}

analyzers = {
    "case_insensitive_keyword": {
        'tokenizer': 'keyword',
        'filter': ['lowercase']
    },        
    "integers": {
        'tokenizer': 'integers',
        'filter': ['leading_zero_trim']
    },
    "n_grams": { # char_n_grams
        'tokenizer': 'char_n_grams',
        "filter": ["lowercase"]
    },  
#    "end_n_grams": {
#        'tokenizer': 'keyword',
#        "filter": ["lowercase", "reverse", "my_edgeNGram", "reverse"]
#    },
    'city': {
        "tokenizer": "standard", # TODO: problem with spaces in words
        "filter": ["my_city_keep", "my_city_synonym", "my_length"] # TODO: shingle ?
    } #,
#    'organization': {
#        "tokenizer": "standard",
#        "filter": ["my_org_keep", "my_org_synonym"]
#    },
    
#    'my_french': {
#        'tokenizer': 'standard',
#        "filter": [
#            "my_city_stop",
#            "my_org_stop",
#            "lowercase",
#            
#            "french_elision",
#            "french_stop",
#            "french_keywords",
#            "french_stemmer"
#          ]
#    }
}

index_settings_template = {
    "settings": {
        "analysis": {
            "tokenizer": tokenizers,
            "filter": filters,
            "analyzer": analyzers
        }
    },

    "mappings": {
        "structure": {}
    }
}

def _gen_index_settings(index_settings_template, columns_to_index):
    '''
    Creates the dict to pass to index creation based on the columns_to_index
    
    NB: the default analyzer is keyword
    
    INPUT:
        - columns_to_index. For:
            {
            "nom_lycee": {
                    'french', 'whitespace', 'integers', 'n_grams', 'city'
                    }, 
            "ville": {
                    'french', 'whitespace', 'city'
                    },
            "ID": {}
            }
    '''
    index_settings = copy.deepcopy(index_settings_template)
    
    field_mappings = {
        key: {
            'analyzer': 'case_insensitive_keyword',
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
            'analyzer': 'case_insensitive_keyword',
            'type': 'string'
        }
        for key, values in columns_to_index.items() if not values
    })
                
    index_settings['mappings']['structure']['properties'] = field_mappings
    
    return index_settings

gen_index_settings = lambda columns_to_index: _gen_index_settings(index_settings_template, columns_to_index)