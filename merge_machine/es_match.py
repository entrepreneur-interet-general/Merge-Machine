#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:39:55 2017

@author: m75380

Add extend sorted keys

Add option to label full file (no inference on unlabelled)

multiple query

security 

Discrepancy btw labelling (worse) and matching (better)

Problem with precision when no match found during labelling

--> add (source_id, None) to pairs when none are found

"""

import numpy as np
import pandas as pd

from .helpers import _bulk_search


def es_linker(es, source, params):
    '''
    Link source to reference following instructions in params. Return 
    concatenation of source and reference with the matches found
    
    INPUT:
        source: pandas.DataFrame containing all source items
        params:
            es: Elasticsearch connection
            index_name: name of the Elasticsearch index to fetch from
            query_template: The query template
            threshold: minimum value of score for this query_template for a match
            must: terms to filter by field (AND: will include ONLY IF ALL are in text)
            must_not: terms to exclude by field from search (OR: will exclude if ANY is found)
            exact_pairs: list of (source_id, ES_ref_id) which are certain matches
    '''
    
    #es = Elasticsearch(**params['es_conn'])
    
    index_name = params['index_name']
    query_template = params['query_template']
    must_filters = params.get('must', {})
    must_not_filters = params.get('must_not', {})
    threshold = params['thresh']
    exact_pairs = params.get('exact_pairs', [])
    non_matching_pairs = params.get('non_matching_pairs', [])
    
    exact_source_indexes = [x[0] for x in exact_pairs if x[1] is not None]
    exact_ref_indexes = [x[1] for x in exact_pairs if x[1] is not None]
    source_indexes = [x[0] for x in source.iterrows() if x [0] not in exact_source_indexes]
    
    def _is_match(f_r, threshold):
        return bool(f_r['hits']['hits']) and (f_r['hits']['max_score'] >= threshold)
    
    def _has_match(f_r):
        return bool(f_r['hits']['hits'])
    
    # Perform matching on non-exact pairs (not labelled)
    if source_indexes:
        rows = (x[1] for x in source.iterrows() if x[0] in source_indexes)
        all_search_templates, full_responses = _bulk_search(es, index_name, [query_template], rows, must_filters, must_not_filters, num_results=1)
        full_responses = [full_responses[i] for i in range(len(full_responses))] # Don't use items to preserve order

        # TODO: remove threshold condition
        matches_in_ref = pd.DataFrame([f_r['hits']['hits'][0]['_source'] \
                                   if _has_match(f_r) \
                                   else {} \
                                   for f_r in full_responses], index=source_indexes)
                        
        ref_id = pd.Series([f_r['hits']['hits'][0]['_id'] \
                                if _has_match(f_r) \
                                else np.nan \
                                for f_r in full_responses], index=matches_in_ref.index)
    
        confidence = pd.Series([f_r['hits']['hits'][0]['_score'] \
                                if _has_match(f_r) \
                                else np.nan \
                                for f_r in full_responses], index=matches_in_ref.index)
        
        confidence_gap = pd.Series([f_r['hits']['hits'][0]['_score'] - f_r['hits']['hits'][1]['_score']
                                if (len(f_r['hits']['hits']) >= 2) and _has_match(f_r) \
                                else np.nan \
                                for f_r in full_responses], index=matches_in_ref.index)

        matches_in_ref.columns = [x + '__REF' for x in matches_in_ref.columns]
        matches_in_ref['__IS_MATCH'] = confidence >= threshold
        matches_in_ref['__ID_REF'] = ref_id
        matches_in_ref['__CONFIDENCE'] = confidence    
        matches_in_ref['__GAP'] = confidence_gap
        matches_in_ref['__GAP_RATIO'] = confidence_gap / confidence

        # Put confidence to zero for user labelled negative pairs
        sel = [x in non_matching_pairs for x in zip(source_indexes, matches_in_ref.__ID_REF)]
        for col in ['__CONFIDENCE', '__GAP', '__GAP_RATIO']:
            matches_in_ref.loc[sel, '__CONFIDENCE'] = 0    
            
    else:
        matches_in_ref = pd.DataFrame()
        
    # Perform matching exact (labelled) pairs
    if exact_ref_indexes:
        full_responses = [es.get(index_name, ref_idx) for ref_idx in exact_ref_indexes]
        exact_matches_in_ref = pd.DataFrame([f_r['_source'] for f_r in full_responses], 
                                            index=exact_source_indexes)
        exact_matches_in_ref.columns = [x + '__REF' for x in exact_matches_in_ref.columns]
        exact_matches_in_ref['__ID_REF'] = exact_ref_indexes
        exact_matches_in_ref['__CONFIDENCE'] = 999
    else:
        exact_matches_in_ref = pd.DataFrame()
    
    #
    assert len(exact_matches_in_ref) + len(matches_in_ref) == len(source)
    new_source = pd.concat([source, pd.concat([matches_in_ref, exact_matches_in_ref])], 1)        
    
    return new_source


    

        
