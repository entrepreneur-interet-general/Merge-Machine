#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from elasticsearch import Elasticsearch, client
import pandas as pd

from merge_machine import es_insert
from merge_machine.es_labeller import ConsoleLabeller
from merge_machine.es_match import es_linker

from merge_machine.es_config import ANALYZERS, DEFAULT_ANALYZER, INDEX_SETTINGS_TEMPLATE

# =============================================================================
# 0. TUTORIAL DATA CAN BE FOUND HERE:
## https://github.com/entrepreneur-interet-general/Merge-Machine/tree/master/examples/data_1
## =============================================================================

# =============================================================================
# 1. USER CONFIG
# =============================================================================

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Choose the files to match, name the reference table and load the source
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Path to the source (dirty) file
source_file_path = 'data_1/source.csv'

# Path to the ref (reference / clean) file
ref_file_path = 'data_1/ref.csv'

# The index name of the reference table in Elasticsearch (if not existant, 
# it will be created)
ref_table_name = 'test_reference'

# -----------------------------------------------------------------------------
# # How to use source ?
#
# How to load the table ?
# See the `pandas` documentation to load table
#
# How many rows to load ? 
# For labelling taking 1000 random rows is sufficient
#
# NB: always use the `dtype=str` option for this package to work
# -----------------------------------------------------------------------------

# Pandas DataFrame containing the source data. 
source = pd.read_csv(source_file_path, 
                    sep=',', encoding='utf-8',
                    dtype=str, nrows=1000)

# For the meantime, do this... #TODO: move this
source = source.where(source.notnull(), '')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Choose the column associations to use for linking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# -----------------------------------------------------------------------------
# # How to use match_cols ?
# EX.1: Adding {'source': 'col1', 'ref': 'colA'} will mean the linking will 
# be looking for similar values to col1 of the source in colA of the reference
# 
# EX.2: Adding {'source': 'col1', 'ref': ['colA', 'colB'] will mean the linking 
# will be looking for similar values to col1 of the source in either colA 
# or colB of the reference
#
# EX.3: Adding {'source': ['col1', 'col2'], 'ref': 'colA' will mean the linking 
# will be looking for similar values to the concatenation of col1 and col2 
# of the source in colA of the reference
# -----------------------------------------------------------------------------

match_cols = [{'source': 'commune', 
               'ref': 'localite_acheminement_uai'},
              {'source': 'Lycées sources', 
               'ref': ('denomination_principale_uai', 'patronyme_uai')},
              {'source': 'département', 
               'ref': 'departement'}]    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the columns to index
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# -----------------------------------------------------------------------------
# # How to use columns_to_index?
#
# The keys of this dictionnary are columns of the reference table. The values
# are elasticsearch analyzers that will be used to compare values being matched
# with these columns. Leaving an empty set will use the default case-indeferent
# keyword analyser. All columns mentioned in columns_to_index will be referenced
# with the default analyzer.
# 
# NB.1: all the columns used in match_cols for the reference table must be
# included in columns_to_index
#
# NB.2: Some of the analyzers used below are custom and their definition will 
# have to be passed during index creation for them to work
# -----------------------------------------------------------------------------


columns_to_index = {
    'departement': {
        'n_grams', 'integers'
    },
    'localite_acheminement_uai': {
        'french', 'n_grams',
    },
    'denomination_principale_uai': {
        'french', 'integers', 'n_grams',
    },
    'patronyme_uai': {
        'french', 'integers', 'n_grams',
    }
}

# -----------------------------------------------------------------------------
# # How to use default_analyzer?
#
# The default analyzer will be used on all columns being indexed.
# -----------------------------------------------------------------------------

default_analyzer = 'case_insensitive_keyword'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configure the Elasticsearch connection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This should work in most cases
es = Elasticsearch(timeout=60, max_retries=10, retry_on_timeout=True)
ic = client.IndicesClient(es)

# =============================================================================
# 2. Index the referential
# =============================================================================

force_re_index = True # Usually set to false

# Create the index
es_insert.create_index(es, ref_table_name, columns_to_index, 
                       default_analyzer=default_analyzer,
                       analyzer_definitions=ANALYZERS,
                       force=force_re_index)

# Insert documents in the index
ref_gen = pd.read_csv(ref_file_path, 
          usecols=columns_to_index.keys(),
          dtype=str, chunksize=40000)
es_insert.index(es, ref_gen, ref_table_name, testing=True)
    
# =============================================================================
# 3. Initiate the labeller
# =============================================================================

# -----------------------------------------------------------------------------
# NB.1:
# Enter `h` or `help` in 
# 
# NB.2: 
# Advanced users may want to skip the labelling process and go directly to 
# linking (step 6.) and enter custom parameters instead of learning them
#
# EX.1:
# For the provide example it might be usefull to add the following filters (=f)
# > must_not_filters / denomination_principale_uai / ['farpi', 'emop', 'employer', 'section', 'greta', 'ctre']
    
# -----------------------------------------------------------------------------

labeller = ConsoleLabeller(es, source, ref_table_name, match_cols, columns_to_index)

# =============================================================================
# 4. Perform the actual labelling
# =============================================================================

labeller.console_labeller()

# =============================================================================
# 5. Print expected performance
# =============================================================================

best_query = labeller.current_queries[0]

print('*** Summary for the best query template (highest score) ***\n')
print('Best query template:\n', best_query._as_tuple(), '\n')
print('Expected precision:', best_query.precision)
print('Expected recall:', best_query.recall)
print('Expected score:', best_query.score)


# =============================================================================
# 6. After labelling, perform the actual linking
# =============================================================================

# -----------------------------------------------------------------------------
# NB.1: 
# Re-load the source to include ALL rows, if necessary (remember we used nrows)

# NB.2:
# If the source is rather large, re-load source using the chunksize option
# and use es_linker on each separate chunk
# -----------------------------------------------------------------------------

new_source = es_linker(es, source, labeller.export_best_params())

# =============================================================================
# 7. Display top results of link
# =============================================================================

for (i, row) in new_source.iloc[:20].iterrows():
    print('*'*50)
    for match in match_cols:
        if isinstance(match['source'], str):
            match['source'] = [match['source']]
        for col in match['source']:
            print(col, '->', row[col])
            
        if isinstance(match['ref'], str):
            match['ref'] = [match['ref']]
        for col in match['ref']:
            print(col, '->', row[col + '__REF'])
        print('\n')
    print('^ is match: {0}; confidence: {1}'.format(row['__IS_MATCH'], row['__CONFIDENCE']))



