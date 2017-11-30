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

import unidecode

def deduplicate(tab, columns):
    '''Creates a smaller version of the original table'''
    
    raise NotImplementedError
    return smaller_tab, indexes


def re_duplicate(tab, smaller_tab, columns, indexes):
    '''
    Takes the output of deduplicate and recreates the table with the original
    size.
    '''
    raise NotImplementedError
    return original_tab

def my_unidecode(string):
    '''unidecode or return empty string'''
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''

def _remove_words(string, words):
    # Replace this by python equivalent of each analyzer ?
    string = my_unidecode(string).lower()
    for word in words:
        string = string.replace(word, '')
    return string

def _reformat_s_q_t(s_q_t):
    '''
    Makes sure s_q_t[1] is a list and s_q_t[2] is list (multi_match) or string
    (match) if there is only one column    
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
    '''
    Generate the string to pass to Elastic search for it to execute query
    
    INPUT:
        - query_template: ((bool_lvl, source_col, ref_col, analyzer_suffix, boost), ...)
        - row: pandas.Series from the source object
        - must: terms to filter by field (AND: will include ONLY IF ALL are in text)
        - must_not: terms to exclude by field from search (OR: will exclude if ANY is found)
        - num_results: Max number of results for the query
    
    OUTPUT:
        - body: the query as string
    
    NB: s_q_t: single_query_template
        source_val = row[s_q_t[1]]
        key = s_q_t[2] + s_q_t[3]
        boost = s_q_t[4]
    '''
    DEFAULT_FILTER_FIELD = '.french' # TODO: replace by standard or whitespace
    
    query_template = [_reformat_s_q_t(s_q_t) for s_q_t in query_template]

    body = {
          'size': num_results,
          'query': {
            'bool': dict({
               must_or_should: [
                          {'match': {
                                  s_q_t[2] + s_q_t[3]: {'query': _remove_words(' '.join(row[idx] for idx in s_q_t[1] if isinstance(row[idx], str)), must_filters.get(s_q_t[2], [])),
                                                        'boost': s_q_t[4]}}
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and isinstance(s_q_t[2], str)
                        ] \
    
                        + [
                          {'multi_match': {
                                  'fields': [col + s_q_t[3] for col in s_q_t[2]], 
                                  'query': _remove_words(row[s_q_t[1]].str.cat(sep=' '), []),
                                  'boost': s_q_t[4]
                                  }
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and (isinstance(s_q_t[2], tuple) or isinstance(s_q_t[2], list))
                        ] \
                for must_or_should in ['must', 'should']
                },
                    **{
                       'must_not': [{'match_phrase': {field + DEFAULT_FILTER_FIELD: {'query': value}}
                                 } for field, values in must_not_filters.items() for value in values],
                       'filter': [{'match_phrase': {field + DEFAULT_FILTER_FIELD: {'query': value}}
                                 } for field, values in must_filters.items() for value in values],
                    })               
                  }
           }

    return body


def _gen_bulk(index_name, search_templates, must, must_not, num_results, chunk_size=100):
    '''
    Create a bulk generator with all search templates
    
    INPUT:
        - search_templates: iterator of form ((query_template, row), ...)
        - num_results: max num results per individual query
        - chunk_size: number of queries per bulk
    
    OUTPUT:
        - bulk_body: string containing queries formated for ES
        - queries: list of queries
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
    '''
    Searches for the values in rows with all the search templates in 
    all_query_templates. Retry on error.
    
    INPUT:
        - all_query_templates: iterator of queries to perform
        - rows: iterator of pandas.Series containing the rows to match
        - num_results: max number of results per individual query    
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
            responses.extend(es.msearch(bulk_body)['responses']) #, index=index_name)
            
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


def _gen_index_settings_from_analyzers(analyzers):
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

 
def gen_index_settings(default_analyzer, columns_to_index, analyzer_index_settings=None):
    '''
    Generate the dictionnary with ES syntax to pass for index creation.
    
    INPUT:
        - default_analyzer: the analyzer that will be used on all fields mentioned
            in columns_to_index no matter what
        - columns_to_index: the analyzers to use for each column
        - analyzer_index_settings: (optional) Elasticsearch settings if to define custom 
                    analyzers if any are used. See Elasticsearch documentation 
                    on how to create custom analyzers
    '''
    # 
    if analyzer_index_settings is not None:
        index_settings = copy.deepcopy(analyzer_index_settings)
    else:
        index_settings = dict()
    
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


def _key_val_er(dict_):
    '''To solve problem with keys having to be strings in json'''
    return [{'__KEY__': key, '__VAL__': value} for key, value in dict_.items()]

def _un_key_val_er(list_):
    def _list_to_tuple(x):
        if isinstance(x, list):
            return tuple(x)
        else:
            return x
    
    try:
        return {_list_to_tuple(dict_['__KEY__']): dict_['__VAL__'] for dict_ in list_}
    except:
        import pdb; pdb.set_trace()
    