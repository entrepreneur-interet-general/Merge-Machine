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



def _is_match(resp, thresh):
    '''Return whether of not the response has hits and best result has a score
    above `thresh`.
    '''
    return bool(resp['hits']['hits']) and (resp['hits']['max_score'] >= thresh)

def _has_hits(resp):
    return bool(resp['hits']['hits'])

def _best_match(list_of_responses, list_of_thresholds):
    '''Return the first real match (above threshold) and if not return the first 
    non empty result if possible. 
    
    Parameters
    ----------
        list_of_responses: list 
        list_of_thresholds: list
    '''
    assert len(list_of_responses) == len(list_of_thresholds)
    for i, (resp, thresh) in enumerate(zip(list_of_responses, list_of_thresholds)):
        if _is_match(resp, thresh):
            return i, resp
    for i, resp in enumerate(list_of_responses):
        if _has_hits(resp):
            return i, resp
    return 0, list_of_responses[0]

def _confidence_estimator(res_of_bulk_search, num_queries, estimator='median'):
    """Look at all results in res_of bulk search apply the estimator (median or
    mean to the confidence associated to each query templpate).
    
    This function is used to scale Elasticsearch confidence scores as their 
    signification varies according to queries.
    """
    
    res_of_bulk_search = [res_of_bulk_search[i] for i in range(len(res_of_bulk_search))] # Don't use items to preserve order
    
    n_r_s = int(len(res_of_bulk_search) / num_queries) # num_rows_source
    
    results = []
    for i in range(num_queries):
        scores = [x['_score'] for res in res_of_bulk_search[i*n_r_s:(i+1)*n_r_s] \
                  for x in res['hits']['hits']]
        results.append(np.__dict__[estimator](scores))
    return results


def _bulk_search_to_full_response(res_of_bulk_search, list_of_thresholds):
    """Take the output _bulk_search and transform it to associate a single 
    result to each row.
    
    Applies `_best_match` to the list of results for each row. This is meant
    to be used in es_linker.
    
    
    Parameters
    ----------
    full_responses: list of Elasticsearch responses
        Flat list of results for search for rows x query_templates.
        Ex: [res_row1_qA, res_row1_qB, res_row2_qA, res_row2_qB, ...]
    list_of_thresholds: list of float
        Thresholds associated to each query.
    
    Returns
    -------
    new_full_responses: list of Elasticsearch responses
        List of responses with the best response for each row, with the same 
        number of items as original rows.
    """
    
    num_rows_source = int(len(res_of_bulk_search) / len(list_of_thresholds))
    assert len(res_of_bulk_search) % num_rows_source == 0
    
    res_of_bulk_search = [res_of_bulk_search[i] for i in range(len(res_of_bulk_search))] # Don't use items to preserve order
    return [_best_match(res_of_bulk_search[n::num_rows_source], list_of_thresholds) \
                      for n in range(0, num_rows_source)]


def _deduplicate(tab, columns, min_diff_prop=0):
    '''Deduplicated the original table according to its values in `columns`.
    tab = pd.DataFrame([[1, 2, 3], [1, 2, 4], [2, 3, 4], [1, 2, 5], [2, 3, 6]], columns=['a', 'z', 'e'])
    
    Parameters
    ----------
    tab: `pd.DataFrame`
    columns: list
        The list of columns to deduplicate on.
    min_diff_prop: float between 0 and 1
        The minimum relative difference between the sizes of the duplicated and
        unduplicated tables for which to perform deduplication.
        
    Returns
    -------
    small_tab: `pd.DataFrame`
        A table where only the first copy of each duplicate is returned. Index
        is preserved from the original table.
    indices: dict
        For the rows that were left out, assign to the index of the row the 
        index of the duplicate row that was kept.
    '''
    
    assert 0 <= min_diff_prop <= 1
    columns = list(set(columns))
    
    small_tab = pd.DataFrame(columns=tab.columns)

    grp = tab.fillna('').groupby(columns)
    print('Len og_tab: {0}; Len deduped: {1}'.format(len(tab), len(grp)))
    
    if len(grp) >= len(tab) * (1-min_diff_prop):
        return None, None
        
    indices = {}
    for _, grp_tab in grp:
        small_tab = small_tab.append([grp_tab.iloc[0]])
        indices.update({idx: grp_tab.index[0] for idx in grp_tab.index})
    
    return small_tab, indices
        

def _re_duplicate(tab, small_tab, indices):
    '''Takes the output of `deduplicate` and recreates the table of the original
    size.
    '''
    # Restrict the new table to the new columns only
    cols = [col for col in small_tab.columns if col not in tab.columns]
    small_tab = small_tab[cols]
    
    join_col = '__SOURCE_GROUP'
    
    tab[join_col] = [indices[idx] for idx in tab.index]
    tab = tab.merge(small_tab, left_on=join_col, right_index=True)

    
    return tab

def _tuplify(obj):
    """Put a string in a tuple or just return object."""
    if isinstance(obj, str):
        return (obj, )
    else:
        return obj


def _priority_bulk_search(es, index_name, all_queries, rows, must_filters, must_not_filters, num_results=3):
    '''Search for multiple rows with multiple query templates returning results
    only for the first query that yields a match.
    
    Bulk search for all rows in `rows` trying -for each row- all query 
    templates in `all_query_templates`. We try all queries one after the other;
    for each row, once a query returns a valid result (with an ES score above 
    the threshold for this query), results for the following queries will be set 
    to None. The output format is the same as that of bulk search.
    
    Parameters
    ----------
    es: instance of `Elasticsearch`
        Connection to Elasticsearch.
    index_name: str
        Name of the index in Elasticsearch.
    all_queries: iterator dict of shape {'best_thresh': XXX, 'template': query_template}
        The query templates to use for search and the threshold above which
        a result will be considered a match.
    rows: iterator of pandas.Series or dict like `{col1: val1, col2: val2, ...}`
        The rows to search for.
    must_filters: `dict` shaped as {column: list_of_words, ...}
        Terms to filter by field (AND: will include ONLY IF ALL are in text).
    must_not_filters: `dict` shaped as {column: list_of_words, ...}
        Terms to exclude by field from search (OR: will exclude if ANY is found
    num_results: int
        The maximum number results per individual query. 
        
    Returns
    -------
    og_search_templates: list
        The search templates that were used in the same order as 
        `full_responses`.
    full_responses: dict containing Elasticsearch results
        Dict indexed by integers. Containing the results of queries in 
        `og_search_template`. The indices correspond to the position of the 
        query in `og_search_templates` (Ex: `full_responses[4]` is the result 
        when searching for `og_search_templates[4]`).
    '''
    import itertools
    
    full_responses = dict()
    all_queries = list(all_queries)
    rows = {i: row for i, row in enumerate(rows)}
    num_rows = len(rows)
    og_search_templates = list(enumerate(itertools.product(all_queries, rows)))
    
    full_responses = dict()
    for i_query, q in enumerate(all_queries):
        print('Len rows at query {0} is {1}'.format(i_query, len(rows)))
        partial_rows = [x[1] for x in sorted(rows.items(), key=lambda x: x[0])]
        partial_idxs = [x[0] for x in sorted(rows.items(), key=lambda x: x[0])] # Og row id
        _, partial_response = _bulk_search(es, index_name, 
                        [q['template']], partial_rows, must_filters, must_not_filters, num_results=1)
        
        # Put as: row_id: response
        partial_response = {partial_idxs[i]: resp for i, resp in partial_response.items()}   
 
        # Add to full response
        full_responses.update({i_query*num_rows + i_row: \
                            val for i_row, val in partial_response.items()})
        
        rows = {i: row for i, row in rows.items() if not _is_match(partial_response[i], q['best_thresh'])}
        
        
    full_responses = {i: full_responses.get(i, {'hits': {'hits': []}}) for i in range(len(og_search_templates))}
    
    return og_search_templates, full_responses

def es_linker(es, source, params):
    """Link source to reference following instructions in params and return the
    concatenation of source and reference with the matches found.
    
    Parameters
    ----------
    source: instance of `pandas.DataFrame`
        The table containing data of the source (dirty data).
    params: dict
        The params dictionnary can be generated by `export_best_params` in the 
        `Labeler`. 
        
        Expected fields are:
            es: instance of `Elasticsearch`
            index_name: str
                name of the Elasticsearch index to fetch from
            queries: list of dicts describing queries.
                The query templates to use to perform matching. Their order determins
                the priority.
            must: dict (optional)
                Terms to filter by field (AND: will include ONLY IF ALL are in text).
            must_not: dict (optional)
                Terms to exclude by field from search (OR: will exclude if ANY is found).
            exact_pairs: list of tuples
                list of (source_id, ES_ref_id) which are certain matches.
                
    Returns
    -------
    new_source: instance of `pandas.DataFrame`
        The result of matching. A DataFrame with the same number of rows as 
        `source` which is the concatenation of `source` and of the matches found 
        in the reference table (columns with suffix: '__REF').
        
        In addition to matching, the following columns are created:
            __IS_MATCH: bool
                Weather or not the result is thought to be a match (the score
                is above the threshold)
            __ID_REF: str
                The document ID of the reference row in the Elasticsearch index
            __ID_QUERY: int
                The place of the query that was used to generate the match in 
                `params['queries']`. The query is chosen using `_best_match`
            __ES_SCORE: float
                The Elascticsearch score returned for the chosen query.
            __THRESH: float
                The threshold associated to the chosen query
            __CONFIDENCE: float
                A score meant to be uniform accross queries. It is computed as 
                follows:
                    scaled_conf = 1 + (conf_q - thresh_q) / mean(all_conf_q)
                A scaled score above 1 indicates a probable match.
            
    """
    
    index_name = params['index_name']
    queries = params['queries'][:3] # queries is {'template': ..., 'threshold':...} / Max 9 queries
    must_filters = params.get('must', {})
    must_not_filters = params.get('must_not', {})
    exact_pairs = params.get('exact_pairs', [])
    non_matching_pairs = params.get('non_matching_pairs', [])
    
    # Deduplicate source
    columns_for_dedupe = [col for q in queries for t in q['template'] for col in _tuplify(t[1])]
    small_source, duplicate_indices = _deduplicate(source, columns_for_dedupe, min_diff_prop=0.1)
    if duplicate_indices is None:
        print('No duplicates found')
        small_source = source
        del source
    
    exact_source_indices = [x[0] for x in exact_pairs if x[1] is not None if x[0] in small_source.index]
    exact_ref_indices = [x[1] for x in exact_pairs if x[1] is not None if x[0] in small_source.index]
    source_indices = [x[0] for x in small_source.iterrows() if x [0] not in exact_source_indices]
    
    # Perform matching on non-exact pairs (not labelled)
    if source_indices:
        rows = (x[1] for x in small_source.iterrows() if x[0] in source_indices)
        
        all_search_templates, res_of_bulk_search = _priority_bulk_search(es, index_name, 
                    queries, rows, must_filters, must_not_filters, num_results=1)
        
        confidence_means = _confidence_estimator(res_of_bulk_search, len(queries), 'mean')

        
        full_responses = _bulk_search_to_full_response(res_of_bulk_search, [q['best_thresh'] for q in queries])
        del res_of_bulk_search

        matches_in_ref = pd.DataFrame([resp['hits']['hits'][0]['_source'] \
                                   if _has_hits(resp) \
                                   else {} \
                                   for _, resp in full_responses], index=source_indices)
                        
        ref_id = pd.Series([resp['hits']['hits'][0]['_id'] \
                                if _has_hits(resp) \
                                else np.nan \
                                for _, resp in full_responses], index=matches_in_ref.index)
    
        confidence = pd.Series([resp['hits']['hits'][0]['_score'] \
                                if _has_hits(resp) \
                                else np.nan \
                                for _, resp in full_responses], index=matches_in_ref.index)
        
        query_index = pd.Series([i for i, resp in full_responses], 
                                index=matches_in_ref.index)
        
        threshold = pd.Series([queries[i]['best_thresh'] \
                                for i, _ in full_responses], index=matches_in_ref.index)

        scaled_confidence = 1 + (confidence - threshold) \
                            / query_index.apply(lambda idx: confidence_means[idx])

        matches_in_ref.columns = [x + '__REF' for x in matches_in_ref.columns]
        matches_in_ref['__IS_MATCH'] = confidence >= threshold
        matches_in_ref['__ID_REF'] = ref_id
        matches_in_ref['__ID_QUERY'] = query_index
        matches_in_ref['__ES_SCORE'] = confidence
        matches_in_ref['__THRESH'] = threshold
        matches_in_ref['__CONFIDENCE'] = scaled_confidence

        
        # TODO: confidence_gap makes no sense with multiple queries
        #        confidence_gap = pd.Series([resp['hits']['hits'][0]['_score'] - resp['hits']['hits'][1]['_score']
        #                                if (len(resp['hits']['hits']) >= 2) and _has_hits(resp) \
        #                                else np.nan \
        #                                for i, resp in full_responses], index=matches_in_ref.index)        
        #        matches_in_ref['__GAP'] = confidence_gap
        #        matches_in_ref['__GAP_RATIO'] = confidence_gap / confidence
        
        # Put confidence to zero for user labelled negative pairs
        sel = [x in non_matching_pairs for x in zip(source_indices, matches_in_ref.__ID_REF)]
        for col in ['__ES_SCORE']: #, '__GAP', '__GAP_RATIO']:
            matches_in_ref.loc[sel, '__ES_SCORE'] = 0
            matches_in_ref.loc[sel, '__CONFIDENCE'] = 0
        
    else:
        matches_in_ref = pd.DataFrame()
    
    # Perform matching exact (labelled) pairs
    print('Num exact ref indices: {0}'.format(len(exact_ref_indices)))
    if exact_ref_indices:
        print('Fetching exact matches')
        full_responses = [es.get(index_name, 'structure', ref_idx) for ref_idx in exact_ref_indices]
        print('Done fetching exact matches')
        exact_matches_in_ref = pd.DataFrame([resp['_source'] for resp in full_responses], 
                                            index=exact_source_indices)
        exact_matches_in_ref.columns = [x + '__REF' for x in exact_matches_in_ref.columns]
        exact_matches_in_ref['__ID_REF'] = exact_ref_indices
        exact_matches_in_ref['__ES_SCORE'] = 999
        exact_matches_in_ref['__CONFIDENCE'] = 999
        
    else:
        exact_matches_in_ref = pd.DataFrame()
    
    #
    assert len(exact_matches_in_ref) + len(matches_in_ref) == len(small_source)
    new_source = pd.concat([small_source, pd.concat([matches_in_ref, exact_matches_in_ref])], 1)        
    
    # Re-create original file if necessary
    if duplicate_indices is not None:
        return _re_duplicate(source, new_source, duplicate_indices)
    
    # Always present source in same order
    special_cols = ['__CONFIDENCE', '__ES_SCORE', '__ID_QUERY', '__ID_REF', 
                       '__IS_MATCH', '__THRESH']
    new_source = new_source[[x for x in new_source.columns if x not in special_cols] \
                            + special_cols]
    
    return new_source

    

        
