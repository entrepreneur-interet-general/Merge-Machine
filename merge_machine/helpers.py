#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 00:38:16 2017

@author: leo

Various helping functions used in the code

"""
import copy
import itertools
import json
import logging

import unidecode

def _my_unidecode(string):
    '''Unidecode or return empty string.'''
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''

def _remove_words(string, words):
    '''Remove words from string.'''
    # TODO: Replace this by python equivalent of each analyzer ?
    string = _my_unidecode(string).lower()
    for word in words:
        string = string.replace(word, '')
    return string

def _reformat_s_q_t(s_q_t):
    '''Makes sure s_q_t[1] is a list and s_q_t[2] is list (multi_match) or 
    string (match) if there is only one column.
    '''
    old_len = len(s_q_t)
    
    col_lists = dict()
    
    #
    if isinstance(s_q_t[1], str):
        col_lists[1] = [s_q_t[1]]
    elif isinstance(s_q_t[1], tuple):
        col_lists[1] = list(s_q_t[1])    
    else:
        col_lists[1] = s_q_t[1]
    assert isinstance(col_lists[1], list) 

    #
    if isinstance(s_q_t[2], tuple):
        if len(s_q_t[2]) == 1:
            col_lists[2] = s_q_t[2][0]
        else:
            col_lists[2] = list(s_q_t[2])    
    else:
        col_lists[2] = s_q_t[2]
    assert isinstance(col_lists[2], list) or isinstance(col_lists[2], str)

    to_return = (s_q_t[0], col_lists[1], col_lists[2], s_q_t[3], s_q_t[4])

    assert len(to_return) == old_len
    return to_return

def _gen_body(query_template, row, must_filters={}, must_not_filters={}, num_results=3):
    '''Generate the dict representation of the json to pass to Elasticsearch 
    for it to execute the desired query.
    
    This function generate the search request body for the data in row using 
    the `query_template`. In addition it can add filters (`must_filters`) or 
    words to exclude (`must_not_filters`).
    
    Parameters
    ----------
    query_template: iterator of tuples of length 5
        Query template represents a compound query template to use in the query.
        (See doc in `query_templates.CompoundQueryTemplate`).
        Ex: ((bool_lvl, source_col, ref_col, analyzer_suffix, boost), ...)
    row: `pandas.Series` or dict like {col1: val, col2: val2, ...}
        The data to search for.
    must_filters: `dict` shaped as {column: list_of_words, ...}
        Terms to filter by field (AND: will include ONLY IF ALL are in text).
    must_not_filters: `dict` shaped as {column: list_of_words, ...}
        Terms to exclude by field from search (OR: will exclude if ANY is found).
    num_results: int
        Maximum number of results for the query.
    
    Returns
    -------
    body: dict
        Dict representation of the JSON object to pass to Elasticsearch to 
        perform the query.

    '''
    
    #==========================================================================
    # NB: 
    # s_q_t: single_query_template
    # source_val = row[s_q_t[1]]
    # key = s_q_t[2] + s_q_t[3]
    # boost = s_q_t[4]
    #==========================================================================
    
    # DEFAULT_FILTER_FIELD = '.standard' # TODO: replace by standard or whitespace
    # If the file is not indexed along this field, the must / must_not filtering will not work
    DEFAULT_FILTER_FIELDS = ['.standard', '.french_estab', '.english'] # Workaround for global filters to work
    
    # CUTOFF_FREQ = 0.001
    
    query_template = [_reformat_s_q_t(s_q_t) for s_q_t in query_template]
    
    body = {
          'size': num_results,
          'query': {
            'bool': dict({
               must_or_should: [
                          {'match': {
                                  s_q_t[2] + s_q_t[3]: {
                                                        'query': _remove_words(' '.join(row[idx] for idx in s_q_t[1] if isinstance(row[idx], str)), must_filters.get(s_q_t[2], [])),
                                                        'boost': s_q_t[4],
                                                        #'cutoff_frequency': CUTOFF_FREQ
                                                        }
                                     }
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and isinstance(s_q_t[2], str)
                        ] \
    
                        + [
                          {'multi_match': {
                                  'fields': [col + s_q_t[3] for col in s_q_t[2]], 
                                  #"type": "best_fields",#"cross_fields",
                                  #"tie_breaker": 0,
                                  'query': _remove_words(' '.join(row[idx] for idx in s_q_t[1] if isinstance(row[idx], str)), []),
                                  'boost': s_q_t[4],
                                  #'cutoff_frequency': CUTOFF_FREQ
                                  }
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and (isinstance(s_q_t[2], tuple) or isinstance(s_q_t[2], list))
                        ] \
                for must_or_should in ['must', 'should']
                },
                    **{
                       'must_not': [{'match': {field + analyzer: {'query': ' '.join(values), 'operator': 'or'}}
                                 } for field, values in must_not_filters.items() if values for analyzer in DEFAULT_FILTER_FIELDS],
                       'filter': [{'match_phrase': {field + analyzer: {'query': value}}
                                 } for field, values in must_filters.items() for value in values for analyzer in DEFAULT_FILTER_FIELDS],
                    })               
                  }
           }
    return body


def _gen_bulk(index_name, search_templates, must, must_not, num_results, chunk_size=1000):
    '''Create a generator of strings for bulk requests in Elasticsearch.
    
    Create bulks of all requests in search_templates and chop them off by 
    `chunk_size` so as not to overload Elasticsearch.
    
    Parameters
    ----------
    index_name: str
        The Elasticsearch index to use for search.
    search_templates: iterator 2 len tuples of shape `(query_template, row)`
        Each tuple represents a query (searching for the data in `row` using
        `query_template`).
    must: `dict` shaped as {column: list_of_words, ...}
        Terms to filter by field (AND: will include ONLY IF ALL are in text).
    must_not: `dict` shaped as {column: list_of_words, ...}
        Terms to exclude by field from search (OR: will exclude if ANY is found).
    num_results: int
        The maximum number results per individual query.
    chunk_size: int
        Number of queries per bulk.
    
    Yields
    -------
    bulk_body: string 
        Bulk query of size `chunk_size` formated for ES.
    queries: list
        The list of the queries performed.
    '''
    
    queries = []
    bulk_body = ''
    i = 0
    for (q_t, row) in search_templates:
        bulk_body += json.dumps({"index" : index_name}) + '\n'
        body = _gen_body(q_t, row, must, must_not, num_results)
        #        if i == 0:
        #            print(body)
        bulk_body += json.dumps(body) + '\n'
        queries.append((q_t, row))
        i += 1
        if i == chunk_size:
            yield bulk_body, queries
            queries = []
            bulk_body = ''
            i = 0
    
    if bulk_body:
        yield bulk_body, queries

def _bulk_search(es, index_name, all_query_templates, rows, must_filters, must_not_filters, num_results=3):
    '''Search for multiple rows with multiple query templates.
    
    Bulk search for all rows in `rows` trying -for each row- all query 
    templates in `all_query_templates`. Results are in the order associated to
    [(query_1, row_1), (query_1, row_2), ...(query_2, row_1), (query_2, row_2), ...]
    
    Parameters
    ----------
    es: instance of `Elasticsearch`
        Connection to Elasticsearch.
    index_name: str
        Name of the index in Elasticsearch.
    all_query_templates: iterator tuples of len 5
        The queries to use for search.
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
    i = 1
    full_responses = dict() 
    og_search_templates = list(enumerate(itertools.product(all_query_templates, rows)))
    search_templates = list(og_search_templates)        
    # search_template is [(id, (query, row)), ...]
    while search_templates:
        # print('At search iteration', i)
        
        bulk_body_gen = _gen_bulk(index_name, [x[1] for x in search_templates], 
                                  must_filters, must_not_filters, num_results)
        responses = []
        for bulk_body, _ in bulk_body_gen:
            logging.debug('Starting bulk search')
            responses.extend(es.msearch(bulk_body)['responses']) #, index=index_name)
            logging.debug('Got {0} results for bulk search'.format(len(responses)))
            
        # TODO: add error on query template with no must or should
        has_error_vect = ['error' in x for x in responses]
        has_hits_vect = [('error' not in x) and bool(x['hits']['hits']) for x in responses]
        
        # Update for valid responses
        for (s_t, res, has_error) in zip(search_templates, responses, has_error_vect):
            if not has_error:
                full_responses[s_t[0]] = res
    
        # print('Num errors:', sum(has_error_vect))
        # print('Num hits', sum(has_hits_vect))
        
        # Limit query to those we couldn't get the first time
        search_templates = [x for x, y in zip(search_templates, has_error_vect) if y]
        i += 1
        
        if i >= 10:
            raise Exception('Problem with elasticsearch: could not perform all queries in 10 trials')
        

    return og_search_templates, full_responses


def _pruned_bulk_search(es, index_name, queries_to_perform, row, must_filters, must_not_filters, num_results):
    """ Performs a smart bulk request, optimized for a large number of 
    queries to perform.
    
    This function is a wrapper around bulk_search that organizes the search
    by branches based on common query cores. It will not search for templates
    if restrictions of these templates already did not return any results.

    Parameters
    ----------
    es: instance of `Elasticsearch`
        Connection to Elasticsearch.
    index_name: str
        Name of the index in Elasticsearch.
    queries_to_perform: list of instances of `CompoundQueryTemplate`
    row: `pandas.Series` or `dict` like {column: value, ...}
        Elements to search for in query.
    must_filters: `dict` shaped as {column: list_of_words, ...}
        Terms to filter by field (AND: will include ONLY IF ALL are in text).
    must_not_filters: `dict` shaped as {column: list_of_words, ...}
        Terms to exclude by field from search (OR: will exclude if ANY is found
    num_results: int
        Maximum number of elements to return using the Elasticsearch query.
    """
    
    results = {}
    core_has_results = dict()
    
    num_queries_performed = 0
    
    sorted_queries = sorted(queries_to_perform, key=lambda x: len(x.core)) # NB: groupby needs sorted 
    for size, group in itertools.groupby(sorted_queries, key=lambda x: len(x.core)):
        size_queries = sorted(group, key=lambda x: x.core)
        
        # 1) Fetch first of all unique cores            
        query_bulk = []
        for core, sub_group in itertools.groupby(size_queries, key=lambda x: x.core):
            # Only add core if all parents can have results
            core_queries = list(sub_group)
            first_query = core_queries[0]
            if all(core_has_results.get(parent_core, True) for parent_core in first_query.parent_cores):
                query_bulk.append(first_query)        
            else:
                core_has_results[core] = False
                
        # Perform actual queries
        num_queries_performed += len(query_bulk)
        query_bulk_tuple = [x._as_tuple() for x in query_bulk]
        search_templates, full_responses = _bulk_search(es, index_name, 
                                                        query_bulk_tuple, [row], 
                                                        must_filters, must_not_filters, num_results)
        bulk_results = [full_responses[i]['hits']['hits'] for (i, _) in search_templates]

        
        # Store results
        results.update(zip(query_bulk, bulk_results))
        for query, res in zip(query_bulk, bulk_results):
            core_has_results[query.core] = bool(res)
            
        # 2) Fetch queries when cores have results
        query_bulk = []
        for core, sub_group in itertools.groupby(size_queries, key=lambda x: x.core):
            if core_has_results[core]:
                query_bulk.extend(list(sub_group))
 
        # Perform actual queries
        num_queries_performed += len(query_bulk)
        query_bulk_tuple = [x._as_tuple() for x in query_bulk]
        search_templates, full_responses = _bulk_search(es, index_name, 
                                                        query_bulk_tuple, [row], 
                                                        must_filters, must_not_filters, num_results)
        bulk_results = [full_responses[i]['hits']['hits'] for (i, _) in search_templates]
        
        # Store results
        results.update(zip(query_bulk, bulk_results))
        
    # Order responses
    to_return = [results.get(query, []) for query in queries_to_perform]
    
    if not sum(bool(x) for x in to_return):
        print('NO RESULTS !!')
    
    print('Num queries before pruning:', len(queries_to_perform))
    print('Num queries performed:', num_queries_performed)
    print('Non empty results:', sum(bool(x) for x in to_return))
    return to_return


def _gen_index_settings_from_analyzers(analyzers):
    '''Takes our custom analyzer definitions and turns them into appropriate
    input for Elasticsearch settings for index creation.
    
    Parameters
    ----------
    analyzers: iterator of dict 
        Iterator of custom definition of analyzers.
        
        Ex: See for example the `city` analyzer in `analyzers_resource.py`
        
    Returns
    -------
    index_settings_template: dict
        A dict formated to be used as input during index creation in 
        Elasticsearch.     
    '''
    index_settings_template = {
        "settings": {
            "analysis": {
                "tokenizer": {},
                "char_filter": {},
                "filter": {},
                "analyzer": {}
            }
        }
    }
    
    for analyzer in analyzers:
        for key in ['tokenizer', 'filter', 'char_filter', 'analyzer']:
            # TODO: check that keys are not overwritten
            index_settings_template['settings']['analysis'][key].update(analyzer.get(key, {}))
    return index_settings_template

 
def gen_index_settings(default_analyzer, columns_to_index, analyzer_index_settings=None):
    '''Generate the to pass to Elasticsearch for index creation.
    
    Parameters
    ----------
    default_analyzer: str
        The analyzer that will be used on all fields mentioned in 
        `columns_to_index` in all cases.
    columns_to_index: dict 
        Keys are the columns to index and values are the Elasticsearch 
        analyzers to use for the corresponding column (in addition to the 
        default analyzer).
            
        Ex: {'col1': {'analyzerA', 'analyzerB'}, 
             'col2': {}, 
             'col3': {'analyzerB'}}
        
        HACK: 
        Passing a string (instead of an iterable) as a value of this dict will 
        change the behavior. The string that is passed will be used as the type
        for the corresponding column (default elasticsearch behavior will apply)

        Ex: {'col1': {'analyzerA', 'analyzerB'}, 
             'numeric_column': 'float'}
        
    analyzer_index_settings: dict or None
        Elasticsearch settings to define custom analyzers if any are used. 
        See Elasticsearch documentation on how to create custom analyzers.
        
    Returns
    -------
    index_settings: dict
        The dict representation of a JSON that can be passed to Elasticsearch
        to create the index corresponding to input.
    '''
    # 
    if analyzer_index_settings is not None:
        index_settings = copy.deepcopy(analyzer_index_settings)
    else:
        index_settings = dict()
    
    # Hack to be able to type columns as non-strings while preserving previous
    # syntax for the library
    columns_to_index_str = {key: val for key, val in columns_to_index.items()
                            if not isinstance(val, str)}
    columns_to_index_typed = {key: val for key, val in columns_to_index.items()
                            if isinstance(val, str)}    
    
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
        for key, values in columns_to_index_str.items() if values
    }
                
    field_mappings.update({
        key: {
            'analyzer': default_analyzer,
            'type': 'string'
        }
        for key, values in columns_to_index_str.items() if not values
    })
                
                
    # Hack: add non-string typed columns
    field_mappings.update({
        key: {
            'type': type_
        }
        for key, type_ in columns_to_index_typed.items()
    })
                
    assert 'mappings' not in index_settings
    index_settings['mappings'] = {'structure': {'properties': field_mappings}}
    
    return index_settings    


def _key_val_er(dict_):
    '''Format a dict as a list to make it JSON serializable.
    
    This deals with dictionnaries for which the type of keys are not valid as 
    json keys but are valid as values. The result is a JSON serializable list.
    This operation can be reversed with `_un_key_val_er`.
    '''
    return [{'K': key, 'V': value} for key, value in dict_.items()] # K for key, V for val

def _un_key_val_er(list_):
    '''Re-creates the original dict from a list created by `_key_val_er`'''
    
    def _list_to_tuple(x):
        if isinstance(x, list):
            return tuple(x)
        else:
            return x
    return {_list_to_tuple(dict_['K']): dict_['V'] for dict_ in list_}
    

def get_header(es, table_name):
    '''Returns the keys of the first element of the table_name index.'''
    return sorted(es.get(table_name, 'structure', 0)['_source'])