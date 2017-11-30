#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:15:15 2017

@author: m75380

Deals with inserting a table in Elasticsearch

"""

import json
import logging
import os
import time

from elasticsearch import Elasticsearch, client
import pandas as pd

from .helpers import gen_index_settings

def pre_process_tab(tab):
    ''' Clean tab before insertion '''
    for x in tab.columns:
        tab.loc[:, x] = tab[x].str.strip()
    return tab


def create_index(es, table_name, columns_to_index, default_analyzer='keyword', analyzer_index_settings=None, force=False):
    '''
    Create a new empty Elasticsearch index (used to host documents)
    
    INPUT:
        - es: an Elasticsearch connection
        - table_name: name of the index in Elasticsearch
        - columns_to_index: dict containing the columns to index and as values
            the analyzers to use in addition to the default analyzer
            
            Ex: {'col1': {'analyzerA', 'analyzerB'}, 
                 'col2': {}, 
                 'col3': 'analyzerB'}
        - force: whether or not to delete and re-create an index if the name 
                is already associated to an existing index
    
    '''
    
    ic = client.IndicesClient(es)
    
    if ic.exists(table_name) and force:
        ic.delete(table_name)
    
    if not ic.exists(table_name):
        index_settings = gen_index_settings(default_analyzer, columns_to_index, analyzer_index_settings)
        try:
            ic.create(table_name, body=json.dumps(index_settings))  
        except Exception as e:
            new_message = e.__str__() + '\n\n(MERGE MACHINE)--> This may be due to ' \
                            'ES resource not being available. ' \
                            'Run es_gen_resource.py (in sudo) for this to work'
            raise Exception(new_message)


def index(es, ref_gen, table_name, testing=False, file_len=0):
    '''
    Insert values from ref_gen in the Elasticsearch index
    
    INPUT:
        - ref_gen: a pandas DataFrame generator (ref_gen=pd.read_csv(file_path, chunksize=XXX))
        - table_name: name of the Elasticsearch index to insert to
        - (testing): whether or not to refresh index at each insertion
        - (file_len): original file len to display estimated time
    '''
    
    ic = client.IndicesClient(es)
    
    # For efficiency, reset refresh interval
    # see https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-update-settings.html
    if not testing:
        low_refresh = {"index" : {"refresh_interval" : "-1"}}
        ic.put_settings(low_refresh, table_name)
    
    # Bulk insert
    logging.info('Started indexing')    
    i = 0
    t_start = time.time()
    for ref_tab in ref_gen:        
        ref_tab = pre_process_tab(ref_tab)
        body = ''
        for key, doc in ref_tab.where(ref_tab.notnull(), None).to_dict('index').items():
            #TODO: make function that limits bulk size
            index_order = json.dumps({
                                "index": {
                                          "_index": table_name, 
                                          "_type": 'structure', 
                                          "_id": str(key)
                                         }
                                })
            body += index_order + '\n'
            body += json.dumps(doc) + '\n'
        es.bulk(body)
        i += len(ref_tab)
        
        # Display progress
        t_cur = time.time()
        eta = (file_len - i) * (t_cur-t_start) / i
        logging.info('Indexed {0} rows / ETA: {1} s'.format(i, eta))

    # Back to default refresh
    if not testing:
        default_refresh = {"index" : {"refresh_interval" : "1s"}}
        ic.put_settings(default_refresh, table_name)        
        es.indices.refresh(index=table_name)



if __name__ == '__main__':
    
    columns_to_index = {
        'SIEGE': {},
        'SIRET': {},
        'SIREN': {},
        'NIC': {},
        'L1_NORMALISEE': {
            'french', 'whitespace', 'integers', 'n_grams', 'city'
        },
        'L4_NORMALISEE': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'L6_NORMALISEE': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'L1_DECLAREE': {
            'french', 'whitespace', 'integers', 'n_grams', 'city'
        },
        'L4_DECLAREE': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'L6_DECLAREE': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'LIBCOM': {
            'french', 'whitespace', 'n_grams', 'city'
        },
        'CEDEX': {},
        'ENSEIGNE': {
            'french', 'whitespace', 'integers', 'n_grams', 'city'
        },
        'NOMEN_LONG': {
            'french', 'whitespace', 'integers', 'n_grams', 'city'
        },
        #Keyword only 'LIBNATETAB': {},
        'LIBAPET': {},
        'PRODEN': {},
        'PRODET': {}
    }
        

    #==============================================================================
    # Index in Elasticsearch 
    #==============================================================================
    testing = False
    force = True
    do_indexing = True
    chunksize = 2000
    
    ref_file_name = 'sirene_filtered.csv' # 'petit_sirene.csv'
    # ref_file_name = 'petit_sirene.csv'
    ref_sep = ',' #';'
    ref_encoding = 'utf-8' #'windows-1252'
    
    if testing:
        nrows = 10000
    else:
        nrows = 10**10
    
    ref_gen = pd.read_csv(os.path.join('local_test_data', 'sirene', ref_file_name), 
                      sep=ref_sep, encoding=ref_encoding,
                      usecols=columns_to_index.keys(),
                      dtype=str, chunksize=chunksize, nrows=nrows) 
    
    
    if testing:
        table_name = '123vivalalgerie4'
    else:
        table_name = '123vivalalgerie2'
    
    es = Elasticsearch(timeout=60, max_retries=10, retry_on_timeout=True)
    
    # https://www.elastic.co/guide/en/elasticsearch/reference/1.4/analysis-edgengram-tokenizer.html

    create_index(es, table_name, columns_to_index, force)

    if do_indexing:
        index(ref_gen, table_name, testing)
