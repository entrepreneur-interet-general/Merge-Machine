#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:59:43 2017

@author: m75380

Custom analyzers for Elasticsearch. No external resource required.
"""


# =============================================================================
# MY FRENCH
# French analyzer with extra knowledge to match establishments
# =============================================================================
french_estab = {
      "filter": {
        "french_elision": {
          "type":         "elision",
          "articles_case": True,
          "articles": [
              "l", "m", "t", "qu", "n", "s",
              "j", "d", "c", "jusqu", "quoiqu",
              "lorsqu", "puisqu"
            ]
        },
        "french_stop": {
          "type":       "stop",
          "stopwords":  "_french_"
        },
        "french_useless": { # words that usually add more noise than anything else
         "type":        "stop",
         "stopwords": ["cedex", "sas", "sarl", "eurl", "sa"]
        },
        "french_abbrev": {
          "type":       "synonym",
          "synonyms":   ['agric, agri => agricole',
                         'agro => agronomique',
                         'assoc, ass, asso => association',
                         'auto, autos, automobiles => automobile',
                         'bat => batiment',
                         'coop => cooperative',
                         'ctre => centre',
                         'grp, groupement => groupe',
                         'copro, coprop, cop, coproprietaires, copr => copropriete',
                         'dep => departement',
                         'dir, directeur => direction',
                         'elec => electrique',
                         'etab => etablissement',
                         'fr, francais => fra',
                         'gen => general',
                         'gym => gymnastique',
                         'immo => immobilier',
                         'indust => industrie',
                         'invest => investissement',
                         'loc => location, local',
                         'lyc => lycee',
                         'med => medical',
                         'music => musique',
                         'nat => national, naturel',
                         'prod => production',
                         'pub, publ, public => publique',
                         'reg => region',
                         'res => residence',
                         'soc => social, societe',
                         'synd, syndic, syndicale => syndicat',
                         'tech => technologie']
        },
        "french_acronyms": {
          "type":       "synonym",
          "synonyms":   ['cnrs => centre national de la recherche scientifique',
                         'inra => institut national de la recherche agronomique',
                         "cea => commissariat à l'énergie atomique et aux énergies alternatives",
                         'inserm => institut national de la santé et de la recherche médicale',
                         'inria => institut national de recherche en informatique et en automatique']           
        },
        "french_stemmer": {
          "type":       "stemmer",
          "language":   "light_french"
        }
      },
      "analyzer": {
        "french_estab": {
          "tokenizer":  "standard",
          "filter": [
            "french_elision",
            "lowercase",
            "french_stop",
            "french_useless",
            "french_abbrev",
            "french_acronyms",
            #"french_keywords",
            "french_stemmer"
          ]
        }
      }
    }


# =============================================================================
# CASE INSENSITIVE KEYWORD ANALYZER
# The analyzer looks for exact match versions in the lowercased text
# =============================================================================

special_keyword = {
     "char_filter": {
             "special_char_filter": {
                "type": "mapping",
                "mappings": ["-=>\\u0020", "é => e", "è => e", "ê => e", "ë => e", "à=>a", "ü=>u"]
                }
             },
     "analyzer": {
            "special_keyword": {
                'tokenizer': 'keyword',
                'char_filter': ['special_char_filter'],
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
        },
        "up_to_5_shingle": {
            	"type": "shingle",
             "token_separator": "",
             "max_shingle_size": 5             
        }
    }, 
        
    'analyzer': {     
        "integers": {
            'tokenizer': 'integers',
            'filter': ['leading_zero_trim', "up_to_5_shingle"]
            }
    }
}
