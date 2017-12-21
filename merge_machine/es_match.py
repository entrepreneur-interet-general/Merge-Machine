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



def _is_match(resp, threshold):
    return bool(resp['hits']['hits']) and (resp['hits']['max_score'] >= threshold)

def _has_hits(resp):
    return bool(resp['hits']['hits'])

def _first_match(list_of_responses, list_of_thresholds):
    '''
    Return the first real match (above threshold) and if not return the first 
    non empty result if possible. 
    
    INPUT: 
        list_of_responses: 
    '''
    assert len(list_of_responses) == len(list_of_thresholds)
    
    for i, resp in enumerate(list_of_responses):
        if _has_hits(resp) and _is_match(resp):
            return resp
    for i, resp in enumerate(list_of_responses):
        if _has_hits(resp):
            return resp
    return 0, list_of_responses[0]
            

def _bulk_search_to_full_response(res_of_bulk_search, list_of_thresholds):
    '''
    Takes the output _bulk_search
    
    
    INPUT:
        full_responses: flat list of results for search for rows x query_templates
            Ex: [res_row1_qA, res_row1_qB, res_row2_qA, res_row2_qB, ...]
        list_of_thresholds: Thresholds associated to each query
    
    OUTPUT:
        new_full_responses: list with the same number of items as original rows
            with each element being the first match found
    '''
    
    num_queries_per_row = len(list_of_thresholds)
    assert len(res_of_bulk_search) % num_queries_per_row == 0
    
    res_of_bulk_search = [res_of_bulk_search[i] for i in range(len(res_of_bulk_search))] # Don't use items to preserve order
    return [_first_match(res_of_bulk_search[n:n+num_queries_per_row], list_of_thresholds) \
                      for n in range(0, len(res_of_bulk_search), num_queries_per_row)]

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
    
    index_name = params['index_name']
    queries = params['queries'] # queries is {'template': ..., 'threshold':...}
    must_filters = params.get('must', {})
    must_not_filters = params.get('must_not', {})
    exact_pairs = params.get('exact_pairs', [])
    non_matching_pairs = params.get('non_matching_pairs', [])
    
    exact_source_indices = [x[0] for x in exact_pairs if x[1] is not None if x[0] in source.index]
    exact_ref_indices = [x[1] for x in exact_pairs if x[1] is not None if x[0] in source.index]
    source_indices = [x[0] for x in source.iterrows() if x [0] not in exact_source_indices]
    
    # Perform matching on non-exact pairs (not labelled)
    if source_indices:
        rows = (x[1] for x in source.iterrows() if x[0] in source_indices)
        all_search_templates, res_of_bulk_search = _bulk_search(es, index_name, 
                    query_templates, rows, must_filters, must_not_filters, num_results=1)
        full_responses = _bulk_search_to_full_response(res_of_bulk_search, thresholds)
        del res_of_bulk_search

        # TODO: remove threshold condition
        matches_in_ref = pd.DataFrame([resp['hits']['hits'][0]['_source'] \
                                   if _has_hits(resp) \
                                   else {} \
                                   for resp in full_responses], index=source_indices)
                        
        ref_id = pd.Series([resp['hits']['hits'][0]['_id'] \
                                if _has_hits(resp) \
                                else np.nan \
                                for resp in full_responses], index=matches_in_ref.index)
    
        confidence = pd.Series([resp['hits']['hits'][0]['_score'] \
                                if _has_hits(resp) \
                                else np.nan \
                                for resp in full_responses], index=matches_in_ref.index)
        
        confidence_gap = pd.Series([resp['hits']['hits'][0]['_score'] - resp['hits']['hits'][1]['_score']
                                if (len(resp['hits']['hits']) >= 2) and _has_hits(resp) \
                                else np.nan \
                                for resp in full_responses], index=matches_in_ref.index)

        matches_in_ref.columns = [x + '__REF' for x in matches_in_ref.columns]
        matches_in_ref['__IS_MATCH'] = confidence >= threshold
        matches_in_ref['__ID_REF'] = ref_id
        matches_in_ref['__CONFIDENCE'] = confidence    
        matches_in_ref['__GAP'] = confidence_gap
        matches_in_ref['__GAP_RATIO'] = confidence_gap / confidence

        # Put confidence to zero for user labelled negative pairs
        sel = [x in non_matching_pairs for x in zip(source_indices, matches_in_ref.__ID_REF)]
        for col in ['__CONFIDENCE', '__GAP', '__GAP_RATIO']:
            matches_in_ref.loc[sel, '__CONFIDENCE'] = 0
        
    else:
        matches_in_ref = pd.DataFrame()
    
    # Perform matching exact (labelled) pairs
    if exact_ref_indices:
        full_responses = [es.get(index_name, ref_idx) for ref_idx in exact_ref_indices]
        exact_matches_in_ref = pd.DataFrame([resp['_source'] for resp in full_responses], 
                                            index=exact_source_indices)
        exact_matches_in_ref.columns = [x + '__REF' for x in exact_matches_in_ref.columns]
        exact_matches_in_ref['__ID_REF'] = exact_ref_indices
        exact_matches_in_ref['__CONFIDENCE'] = 999
        
    else:
        exact_matches_in_ref = pd.DataFrame()
    
    #
    assert len(exact_matches_in_ref) + len(matches_in_ref) == len(source)
    new_source = pd.concat([source, pd.concat([matches_in_ref, exact_matches_in_ref])], 1)        
    
    return new_source


    

        
