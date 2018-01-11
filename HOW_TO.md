# Merge-Machine: How To Make it Work ?

## What can I expect?
This tools helps you link dirty table data with a clean referential using Elasticsearch. No magic is involved. The cleaner the data, the better the matching will be. Basically, if a human with no prior knowledge of the data is able to distinguish a match from a non-match, the machine may succeed in matching the files; if not, it probably won't be very helpful...

#### Assumptions

* It is expected that the entities in the dirty file are a subset of those in the reference file (i.e. it should be possible to find a match for all rows). 
* Matching rows contain have matching words (or almost). The matching is essentially text based and tries to find similarities in letters and words.
* No semantic knowledge is needed. The machine does have human type knowledge and doesn't know, for example, that "Castle" and "Fort" are close and could be used for a match. That being said, there are some cases where semantic knowledge was added using Elasticsearch's synonym function; for example with cities, it is possible to find "Paris" when looking for "Lutece" (Paris's old name)... You can add this sort of knowledge if you can provide some equivalence dataset.
* No external knowledge is needed. The machine will match entities based only on the data that is provided in each row. It will not go fetch external data on some other database (except for synonyms described above).

### Concepts
* `source`: A dirty table for which we are trying to find a reference
* `ref` (referential): A clean table in which we will look for elements of the source
* `query_template`: A custom representation of a scheme of an Elasticsearch query that will be used to search for elements of the source in the reference
* `labeller`: A python object that uses user labelled pairs of matches/non_matches to learn the optimal query template to use for the best matching

### Workflow
1. Load source and reference tables
2. [Choose the columns which should match between the source and reference and which are likely to be useful in distinguishing a match](#2-column-pairing)
3. [Index the referential in Elasticsearch (in particular, index the columns used for matching)](#3-indexing-the-referential-in-elasticsearch)
4. [(Optional) Labelling: learn the optimal parameters for file linking by labelling pairs between the source and referential as match / non-match](#4-labelling--learning-optionnal)
5. [Perform matching using the query templates inputed by the user or learned after labelling](#5-linking)

## 2. Column pairing

Column pairing describes what columns share information that can be used for matching. You can indicate multiple column pairs to indicate that multiple informations can be used (matching "ESTABLISHMENT" and "STATE" columns for example). Also, pairings are not necessaraly one-to-one but can also be one-to-many, many-to-one or many-to-many; the different behaviors are described below.

### Adding multiple pairs
Pairings are indications for the learning algorithm. The algorithm will most likely select a subset of the most useful pairs for matching. The selected pairs will act as "AND" arguments. Although adding as many pairs as possible might be tempting (Who knows, maybe matching "EMAIL" with "COUNTRY" could help?), it will increase the amount of necessary computation and might add noise which could subsequently lower the performance of matching. Ideal pairing is usually the mimimal set of columns so that a human could decide if a result pair is a match or not.

#### Multiple columns for the source in a column pairing (many-to-one)
When multiple columns are selected for the source in a pair; the values of these columns will be concatenated (with a space), and the resulting value will be searched for as one in the referential.

```
For example: {"source": ["NAME", "SURNAME"], "ref": "FULL NAME" }
```

#### Multiple columns for the reference in a column pairing (one-to-many)
When multiple columns are selected for the reference, the value of the source will be searched in the columns of the reference and results can be returned if their are matches for any of these columns; thus acting as a "OR" argument.

```
For example: {"source": "Client Full Name", "ref": ["Maried Name", "Maiden Name"]}
```

#### Multiple columns on both sides (many-to-many)
This simply combines the effects of many-to-one and one-to-many.

```
For example: {"source": ["CLIENT NAME", "CLIENT SURNAME"], "ref": ["Maried Name", "Maiden Name"]}
```

## 3. Indexing the referential in Elasticsearch
Once it is known what columns will be used for matching, we insert our reference file in Elasticsearch while using the appropriate analyzers 

### What is an analyzer?
See [this post](https://stackoverflow.com/a/12846637/7856919) and [the official documentation](http://nocf-www.elastic.co/guide/en/elasticsearch/reference/current/analysis.html). This package uses standard Elasticsearch analyzers (french, ...) as well as custom ones (city, ...); you could easily add your own if necessary. By default, the package uses the following:

- french: Splits the text on words and performs common stemming adapted for french text 
- case_insensitive_keyword: (custom) Case insensitive exact match (Ex: Fields should match)
- n_grams: (custom) Character 3-grams (Ex: When word stemming is not good enough due to spelling mistakes etc.)
- integers: (custom) Extract all integers from text (Ex: to match id's buried in text)
- city: (custom; requires resource) Extract only city names (in any language) and translate to a common language (Ex: Matching city names in different languages) (Ex 2: Extracting city names from an organization to match with geographical information)
- country: (custom; requires resource) Extract only country names (in any language) and translate to a common language

### How to choose the appropriate analyzers
You can choose multiple analyzers per column. Pertinant analyzers will be able to extract distinctive features that are useful for matching. For example, when matching two address fiels, you might want to use 1) The integers analyzer (to get any street number); 2) The french analyzer (to get street names) 3) The city analyzer (to match only results that are in the same city). As for column pairing, increasing the number of analyzers might increase the theoretical ability for matching but will also reduce performance (memory, speed) and might induce noise that will lead to bad learning.

## 4. Labelling / learning (optionnal)
In the labelling phase, the user is asked to inform whether a pair of rows (one from the source, one from the referential) is thought to be a match. Labelling can be used for two purposes: 1) To learn the optimal parameters for linking. 2) To manually link two files (this may be much faster than doing that on excel for example).

### Possible answers
- yes: the pair is a match
- no: the pair is not a match
- uncertain: information is missing to decide (the row will be skipped)
- forget: the row of the source has no match in the referential and should be skipped

### How it works 
The labeller generates a very large amount of query templates (for different combinations of analzyers, column pairs and boosts). For each new line of the source being labelled, it uses these query templates to look for the source row. Once the user labels a pair as a match, it compairs the real match with the results proposed by each query template and then updates the performance of each query template (precsion and recall). For each query template we compute the best threshold on the Elasticsearch score so as to optimize a custom score. Results above that threshold will be considered to be matches. Query templates are then sorted by score. We regularly filter out query templates for which the score or pricision is too low. 

## 5. Linking
You may want to skip learning alltogether and manually input the parameters for linking. Information to specify are the following: `index_name`, `ref_index_name`, `params`, `queries`, `must`, `must_not`.

### Query templates
Query templates are a package-specific way of describing a matching strategy using elasticsearch. Query templates (or compound query templates) describe the combination of analyzers for each column pair, and the boost (see [the official documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-boost.html)) to use. Compound query templates are actually a list of single query templates which each are a 5-tuple looking like this:
```
(bool_lvl, source_cols, ref_cols, analyzer_suffix, boost)
```
- bool_lvl: Either "should" or "must" (see [the official documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html))
- source_cols: the column / columns of the source to use
- ref_cols: the column / columns of the reference in which to look for values from source_cols
- analyzer_suffix: which analyzer to use for matching
- boost: the Elasticsearch boost parameter

### Threshold
The Elasticsearch score above which a result will be considered to be a match.

### Must and Must-not
This allows to set mandatory (must) or forbidden (must_not) keywords for all the matches found in the referential for the entire file. This can be useful when looking for a subset of the reference that is identifiable by a keyword (Ex: add "must": {"ESTABLISHMENT": ["university"]} to force all results to include the word "university" in the "ESTABLISHMENT" field) (Ex 2: add "must_not": {"ESTABLISHMENT": ["alumni"]} to exclude any potential match with the word alumni in the "ESTABLISHMENT" field). This can greatly increase the accuracy of matching in certain cases.

In a one-shot usecase, this might not be useful as the user might have already filtered the reference data. But when matching against a large reference database multiple times, this can avoid useless re-indexing of the file while preserving the increased accuracy of matching against a more accurate subset.

### Queries ? (plurral ?)
Yes! This module actually uses several query templates (and their associated thresholds). If the first one does not yield any result, then the second one will be used etc. This can be useful when the source file is not homogeneous and their are multiple best strategies for matching depending on the line that is being match.
