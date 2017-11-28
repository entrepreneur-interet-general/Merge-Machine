#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:27:30 2017

@author: m75380

A list of standard Elasticsearch analyzers

"""
import copy
import os

#curdir = os.path.dirname(os.path.realpath(__file__))
#os.chdir(curdir)

#city_keep_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_keep.txt')
#city_syn_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_synonyms.txt')

elasticsearch_resource_dir = '/etc/elasticsearch'

organization_keep_file_path = 'es_organization_keep.txt'
organization_syn_file_path = 'es_organization_synonyms.txt'




#
#filters = {
#    "my_edgeNGram": {
#        "type": "edgeNGram",
#        "min_gram": 3,
#        "max_gram": 30
#    },

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




#    "end_n_grams": {
#        'tokenizer': 'keyword',
#        "filter": ["lowercase", "reverse", "my_edgeNGram", "reverse"]
#    },
     #,
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



# =============================================================================
# CASE INSENSITIVE KEYWORD ANALYZER
# The analyzer looks for exact match versions in the lowercased text
# =============================================================================
case_insensitive_keyword = {
     'analyzer': {
            "case_insensitive_keyword": {
                'tokenizer': 'keyword',
                'filter': ['lowercase']
             } 
            }
    }

# =============================================================================
# 3-GRAMS 
# =============================================================================

n_grams = {
    'tokenizer': {    
        "char_n_grams": {
            "type": "ngram",
            "min_gram": 3,
            "max_gram": 3,
            "token_chars": [
                "letter",
                "digit"
            ]
            }
        },
        
    'analyzer': {
        "n_grams": { # char_n_grams
            'tokenizer': 'char_n_grams',
            "filter": ["lowercase"]
            }
        }
    }


# =============================================================================
# INTEGERS ANALYZER
# Extracts distinct integers from text
# =============================================================================

integers = {

    'tokenizer': {
        "integers": {
            "type": "pattern",
            "preserve_original": 0,
            "pattern": '(\\d+)',
            'group': 1
        }
    },     
        
    'filter':{
        "leading_zero_trim": {
            "type": "pattern_replace",
            "pattern": "^0+(.*)",
            "replacement": "$1"
        }
    }, 
        
    'analyzer': {     
        "integers": {
            'tokenizer': 'integers',
            'filter': ['leading_zero_trim']
            }
    }
}

# =============================================================================
# CITY ANALYZER
# The city analyzer is a cateforical analyzer to find mentions of a city 
# regardless of the language.
# The city analyzers filters down the tokenized text to tokens matching
# a fixed list of known cities in various languages and then translates the city
# name to a fixed language (may be different for each city).
# =============================================================================
        
city_keep_file_path = 'es_city_keep.txt'
city_syn_file_path = 'es_city_synonyms.txt'        
        
city = {
     'tokenizer': {
                 "my_standard": { #standard-ish except for "-"
                        "type": "pattern",
                        "pattern": '|'.join(["\'", '\"', '\(', '\)', '_', ',', '\.', ';'])
                    }
             },  
        
     'filter': {
                    "city_keep" : {
                        "type" : "keep",
                        "keep_words_case": True, # Lower the words
                        "keep_words_path" : city_keep_file_path
                    },
                            
                    "city_stop":{
                        "type" : "stop",
                        "ignore_case": True,
                        "stopwords_path": city_keep_file_path      
                    },
                      
                    "city_synonym" : {
                        "type" : "synonym", 
                        "expand": False,    
                        "ignore_case": True,
                        # "synonyms" : ["paris, lutece => paname"],
                        "synonyms_path" : city_syn_file_path,
                        "tokenizer" : "my_standard"  # TODO: whitespace? 
                    },
                            
                    "city_length": {
                        "type" : "length",
                        "min": 4
                    }  
                },
     
     'analyzer': {
             'city': {
                        "tokenizer": "standard", # TODO: problem with spaces in words
                        "filter": ["city_keep", "city_synonym", "city_length"] # TODO: shingle ?
                    }
             } 
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


default_analyzer = 'case_insensitive_keyword'
 
def _gen_index_settings_tmp(default_analyzer, columns_to_index, index_settings=None):
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
    
    
    
def _gen_index_settings(index_settings_template, columns_to_index, default_analyzer):
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

# Create
gen_index_settings = lambda columns_to_index, default_analyzer: \
   _gen_index_settings(index_settings_template, columns_to_index, default_analyzer)

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


