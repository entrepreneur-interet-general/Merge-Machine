#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:59:43 2017

@author: m75380

Custom analyzers for Elasticsearch. No external resource required.
"""

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
        "n_grams_char": {
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
        "n_grams": { # n_grams_char
            'tokenizer': 'n_grams_char',
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
