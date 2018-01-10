
### Concepts

* `source`: A dirty table for which we are trying to find a reference
* `ref` (referential): A clean table in which we will look for elements of the source
* `query_template`: A custom representation of a scheme of an Elasticsearch query that will be used to search for elements of the source in the reference
* `labeller`: A python object that uses user labelled pairs of matches/non_matches to learn the optimal query template to use for the best matching
