# The Magical CSV Merge Machine

## What does it do ?

A python3 library to link a dirty CSV file with a clean reference table. It is meant to as generic as possible and includes a labeller to learn optimal parameters for a usecase.


## How to install ?

### Manual install

#### Non-Python Requirements

This library relies on Elasticsearch. We recommend using the last version available. Instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html).

#### PIP3 install 

```
git clone https://github.com/eig-2017/Merge-Machine
cd Merge-Machine
pip3 install -e .
```

## How to use ?

### Concepts

* `source`: A dirty table for which we are trying to find a reference
* `ref` (referential): A clean table in which we will look for elements of the source
* `query_template`: A custom representation of a scheme of an Elasticsearch query that will be used to search for elements of the source in the reference
* `labeller`: A python object that uses user labelled pairs of matches/non_matches to learn the optimal query template to use for the best matching

### How to use in python3 ? 

See an example in [tests/example.py](https://github.com/eig-2017/Merge-Machine/blob/master/tests/example.py).

## How it works ?

The reference is indexed in Elasticsearch with multiple indexes (languages specific, integers, n\_grams...). The labeller then proposes training samples from the source which it tries to match to rows of the reference file. Upon user confirmation (match / not match) it updates its belief on which Elasticsearch queries are most performant to use for matching. When labelling is over, the "best query" (a weighted combination of multiple ES queries with different analyzers on different fields) is used for each row of the source to try to find a match in the ES-indexed referential.

## How to contribute ?

Feel free to report bugs via Issues and make pull requests...


## Credits

This library was developped during 10 months in 2017 at the French [Ministry of Research and Higher Education](http://www.enseignementsup-recherche.gouv.fr/) in the context of the ["Entrepreneur d'Intérêt Général" program](https://www.etalab.gouv.fr/decouvrez-la-1e-promotion-des-entrepreneurs-dinteret-general) funded by the French Government.

## See also

This library was developped as a component of larger matching service:
* ONLINE SERVICE COMING SOON !!!
* [code](https://github.com/eig-2017/the-magical-csv-merge-machine)

Other similar libraries include:
* [match_id](https://github.com/matchID-project) (Identity record linking)
* [dedupe](https://github.com/dedupeio/dedupe) (Record linking and deduping)
