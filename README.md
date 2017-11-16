# The Magical CSV Merge Machine

## What does it do ?

A python3 library to link a dirty CSV file with a clean reference table. It is meant to as generic as possible and includes a labeller to learn optimal parameters for a usecase.


## How to install ?

### Manual install

#### Non-Python Requirements

This library relies on Elasticsearch. We recommend using the last version available. Instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html).

#### PIP3 install 

```
# TODO: change to real name
pip3 install -e .

```


#### Docker install

Not yet... Sorry :(


## How to use ?

### Concepts

* `source`: A dirty table for which we are trying to find a reference
* `ref` (referential): A clean table in which we will look for elements of the source
* `query_template`: A custom representation of a scheme of an Elasticsearch query that will be used to search for elements of the source in the reference
* `labeller`: A python object that uses user labelled pairs of matches/non_matches to learn the optimal query template to use for the best matching

### How to use in python3 ? 

```
import os

import pandas as pd

import magical_merge_machine as mmm

# Choose the files that will be matched
source_file_path = 'test/test_1/source.csv' # Dirty file
ref_file_path = 'test/test_1/ref.csv' # Reference file



# Indicate what columns should be used for matching
match_cols = {
	
				... Fill here
			}

# Index the reference file in Elasticsearch
ref_index_name = 'index_for_ref' # NB: indexes can be re-used across projects
columns_to_index = mmm.gen_columns_to_index(match_cols)
mmm.create_index()

# (OPTIONAL) Labelling: learn the optimal query_template
labeller = ConsoleLabeller(source, 
					ref_index_name, 
                 	match_cols, 
                 	columns_to_index, 
                 	certain_column_matches=None, 
                 	must={}, 
                 	must_not={})  

## Label pairs (y(es)/n(o)/(p)revious/(q)uit)
labeller.console_labeller()
params = labeller.export_best_params()
   

# If no labelling, define parameters
## params = ...

# Perform linking
new_source = es_linker(source, params)

# Write to file
new_source.to_csv('results.csv')
```

NB: In this example, the entire source is loaded in memory before matching. For large files, we suggest using pandas "chunksize" option to read the file by blocks


## How to contribute ?


## Credits

This library was developped during 10 months in 2017 at the French [Ministry of Research and Higher Education](http://www.enseignementsup-recherche.gouv.fr/) in the context of the ["Entrepreneur d'Intérêt Général" program](https://www.etalab.gouv.fr/decouvrez-la-1e-promotion-des-entrepreneurs-dinteret-general) funded by the French Government.

## See also

This library was developped as a component of larger matching service:
* service url
* github url

Other similar libraries include:
* [match_id](https://github.com/matchID-project) (Identity record linking)
* [dedupe](https://github.com/dedupeio/dedupe) (Record linking and deduping)
