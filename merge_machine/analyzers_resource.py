#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:06:17 2017

@author: m75380

Custom analyzers for Elasticsearch requiring processing of external resources.

"""


#curdir = os.path.dirname(os.path.realpath(__file__))
#os.chdir(curdir)

#city_keep_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_keep.txt')
#city_syn_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_synonyms.txt')

#elasticsearch_resource_dir = '/etc/elasticsearch

#organization_keep_file_path = 'es_organization_keep.txt'
#organization_syn_file_path = 'es_organization_synonyms.txt'




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
#        "tokenizer" : "city_standard"  # TODO: whitespace? 
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
                 "city_standard": { #standard-ish except for "-"
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
                        "tokenizer" : "city_standard"  # TODO: whitespace? 
                    },
                            
                    "city_length": {
                        "type" : "length",
                        "min": 4
                    }  
                },
     
     'analyzer': {
             'city': {
                        "tokenizer": "city_standard", # TODO: problem with spaces in words
                        "filter": ["city_keep", "city_synonym", "city_length"] # TODO: shingle ?
                    }
             } 
     }
