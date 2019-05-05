#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools to label matches and learn the optimal query templates for linking.
"""
from collections import defaultdict
import copy
import hashlib
import itertools
import json
import logging
import random
import time

from elasticsearch import client, RequestError
import numpy as np

from .helpers import _bulk_search, _key_val_er, _pruned_bulk_search, _un_key_val_er
from .my_json_encoder import MyEncoder
from .query_templates import SingleQueryTemplate, CompoundQueryTemplate

# =============================================================================
# Dev decorators
# =============================================================================

INDENT_LVL = 0
TIME_IN = defaultdict(float)

def print_name(function):
    """Print the function being run with indentation for inclusion.
    
    To be used as a decorator.
    """
    def wrapper(*args, **kwargs):
        global INDENT_LVL
        my_id = random.randint(0, 100000)
        ind = 2 * INDENT_LVL * ' '
        INDENT_LVL += 1
        print('{0}<< Starting: {1} ({2})'.format(ind, function.__name__, my_id))
        start_time = time.time()
        res = function(*args, **kwargs)
        INDENT_LVL -= 1
        print('{0}>> Ending: {1} ({2}) / Took {3}s'.format(ind, \
                      function.__name__, my_id, time.time()-start_time))
        return res
    return wrapper

def time_in(function):
    def wrapper(*args, **kwargs):
        global TIME_IN
        start_time = time.time()
        res = function(*args, **kwargs)
        TIME_IN[function.__name__] += time.time()-start_time
        return res
    return wrapper

def _print_time_in():
    global TIME_IN
    print('PRINTING TIME IN:')
    for key, value in TIME_IN.items():
        print('{}: {}s'.format(key, value))

# =============================================================================
# Helper functions (for use in Labeller)
# =============================================================================

def _gen_suffix(columns_to_index, s_q_t_2):
    """Yield suffixes to add to field_names for the given analyzers.
    
    Suffixes (e.g. `.french`, `.standard`, ...) designate the Elasticsearch 
    analyzer that will be used during querying.
    
    Parameters
    ----------
    columns_to_index: dict
        Dictionnary linking each column of the referential to the analyzers 
        used for indexing.
    s_q_t_2: str or tuple
        Column or tuple of columns from the referential.
        NB: s_q_t_2: element two of the single_query_template.
    
    Yields
    ------
    str
        Analyzer suffix.
    """
    
    if isinstance(s_q_t_2, str):
        analyzers = columns_to_index[s_q_t_2]
    elif isinstance(s_q_t_2, tuple):
        analyzers = set.union(*[columns_to_index[col] for col in s_q_t_2])
    else:
        raise ValueError('s_q_t_2 should be str or tuple (not list)')
    yield ".special_keyword" # No suffix for standard analyzer
    for analyzer in analyzers:
        yield '.' + analyzer


def DETUPLIFY_TODO_DELETE(arg):
    if isinstance(arg, tuple) and (len(arg) == 1):
        return arg[0]
    return arg

def _gen_all_single_query_template_tuples(match_cols, columns_to_index, bool_levels,
                                          boost_levels):
    """Generate a list of single query tuples"""
    return list(((bool_lvl, DETUPLIFY_TODO_DELETE(x['source']), DETUPLIFY_TODO_DELETE(x['ref']), suffix, boost) \
                                       for x in match_cols \
                                       for suffix in _gen_suffix(columns_to_index, x['ref']) \
                                       for bool_lvl in bool_levels.get(suffix, ['must']) \
                                       for boost in boost_levels))

def _gen_all_query_template_tuples(match_cols, columns_to_index, bool_levels, 
                            boost_levels, max_num_levels):
    """ 
    Generate query templates (as tuples). This will generate most combinations
    of queries that are possible using the different input.
    
    Parameters
    ----------
    match_cols: dict
        # TODO: see elsewhere
    columns_to_index: dict
        # TODO: see elsewhere
    bool_levels: list of strings
        list of the bool levels to generate on.
        Values can be `['must']`, or `['must', 'should']`.
    boost_levels: list of floats
        All the original boosts (see Elasticsearch boost) to include in 
        the output. It is strongly recommended to initiate with [1] (all boosts
        at 1) to avoid generating too many query templates.
    max_num_levels: int
        The maximal number of single queries in the compound queries that 
        are generated.
                    
    Returns
    -------
    all_query_templates: list of tuples
        Representation of multiple compound query templates.
    """
    single_queries = _gen_all_single_query_template_tuples(match_cols, 
                                    columns_to_index, bool_levels, boost_levels)
    all_query_templates = list(itertools.chain(*[list(itertools.combinations(single_queries, x)) \
                                        for x in range(2, max_num_levels+1)][::-1]))
    # Queries must contain at least two distinct columns (unless one match is mentionned)
    # TODO: check why removing this creates huge computation
    if len(match_cols) > 1:
        all_query_templates = list(filter(lambda query: len(set((x[1], x[2]) for x in query)) >= 2, \
                                    all_query_templates))

    all_query_templates = list(filter(lambda query: 'must' in [s_q_t[0] for s_q_t in query], \
                                all_query_templates))
    return all_query_templates

# =============================================================================
# Labeller Classes
# =============================================================================
        
class LabellerQueryTemplate(CompoundQueryTemplate):
    """
    Extension of CompoundQueryTemplate which can also store data relative
    to the performance of the query in the context of labelling.
    """
    
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.reset()
        
    def _sanity_check(self):
        if self.history_pairs:
            assert all(isinstance((val + [(None,)])[0], tuple) for val in self.history_pairs.values())
        
        assert len(self.history_pairs) \
                == len(self.first_scores) \
                == len(self.has_results)
        
        assert len(self.first_is_match) == len(self.any_is_match)

    def reset(self):
        self.history_pairs = {} # {source_id: [(), (), ]}
        #self.steps = [] # Num rows labelled at which this pair was added (in case of later creation)
        self.first_scores = {} # ES scores of first result of queries
        self.all_scores = {}
        self.has_results = {} # Whether the query returned any result
        self.first_is_match = {} # First hit is the actual match
        self.any_is_match = {} # Any of the hits is the actual match
        
        self.thresh = None
        self.precision = None
        self.recall = None
        self.inclusion_ratio = None
        self.score = 0.1 # General score of the template based on precision and recall        

    def to_dict(self):
        """Returns a dict representation of the instance."""
        dict_ = dict()
        dict_['query'] = super().to_dict()
        dict_['history_pairs'] = _key_val_er(self.history_pairs)
        dict_['first_scores'] = _key_val_er(self.first_scores)
        dict_['all_scores'] = _key_val_er(self.all_scores) # added this
        dict_['has_results'] = _key_val_er(self.has_results)
        dict_['first_is_match'] = _key_val_er(self.first_is_match)
        dict_['any_is_match'] = _key_val_er(self.any_is_match)
        
        dict_['thresh'] = self.thresh
        dict_['precision'] = self.precision
        dict_['recall'] = self.recall
        dict_['inclusion_ratio'] = self.inclusion_ratio
        dict_['score'] = self.score
        return dict_
    
    @classmethod
    def from_dict(cls, dict_):
        """Returns an instance of the class using a representation generated 
        by to_dict.
        """
        lqt = super(LabellerQueryTemplate, cls).from_dict(dict_['query'])
        lqt.history_pairs = {key: [tuple(x) for x in values] for key, values in _un_key_val_er(dict_['history_pairs']).items()}
        lqt.first_scores = _un_key_val_er(dict_['first_scores'])
        lqt.all_scores = _un_key_val_er(dict_['all_scores'])
        lqt.has_results = _un_key_val_er(dict_['has_results'])
        lqt.first_is_match = _un_key_val_er(dict_['first_is_match'])
        lqt.any_is_match = _un_key_val_er(dict_['any_is_match'])
        
        lqt.thresh = dict_['thresh']
        lqt.precision = dict_['precision']
        lqt.recall = dict_['recall']
        lqt.inclusion_ratio = dict_['inclusion_ratio']
        lqt.score = dict_['score']
        return lqt
        
    def add_pairs(self, source_idx, pairs, first_score, all_scores=[]):
        """Add pairs to history_pairs (pairs that were generated by this query).
        
        Parameters
        ----------
        source_idx: same type as source index elements
            The index of the element of the source being treated.
        pairs: list of tuples
            List of tuples (source_idx, ref_idx) that were found as possible 
            matches when searching for `source_idx`.
        first_score: float
            The score associated to the best query result.
        all_scores: list of float
            The scores associated to each pair in `pairs`.
        """
        self.history_pairs[source_idx] = pairs
        self.has_results[source_idx] = bool(pairs)
        self.first_scores[source_idx] = first_score
        self.all_scores[source_idx] = all_scores
        
    
    def _get_score_for_pair(self, source_idx, ref_idx):
        pairs = self.history_pairs[source_idx]
        all_scores = self.all_scores[source_idx]
        assert len(pairs) == len(all_scores)
        score_lookup_dict = {ref_idx: score \
                    for (source_idx, ref_idx), score in zip(pairs, all_scores)}
        return score_lookup_dict[ref_idx]
    
    def add_labelled_pair(self, source_idx, labelled_pair):
        """Update `first_is_match`, `any_is_match` with a labelled pair. 
        
        Use the label indicated in `labelled_pair` to update `first_is_match`
        and `any_is_match` accordingly. Use this after `add_pairs`.
        
        Parameters
        ----------
        source_idx: same type as source index elemnts
            The index of the element of the source being treated.
        labelled_pair: pair of shape (source_idx, label)
            The label follows the convention established in the `Labeller` class        
        """
        
        assert len(self.first_scores) == len(self.history_pairs)
        
        # If row is forgotten or no matches were found in all queries, mark as unaplicable
        if labelled_pair[1] == '__FORGET': 
            self.first_is_match[source_idx] = None # None if does not apply
            self.any_is_match[source_idx] = None # None if does not apply
        elif labelled_pair[1] == '__NO_RESULT': # Consider that no results <-> False
            self.first_is_match[source_idx] = False 
            self.any_is_match[source_idx] = False
        else:
            if self.history_pairs[source_idx]:        
                self.first_is_match[source_idx] = labelled_pair==self.history_pairs[source_idx][0]
                self.any_is_match[source_idx] = labelled_pair in self.history_pairs[source_idx]
            else:
                self.first_is_match[source_idx] = False
                self.any_is_match[source_idx] = False
            
    def unlabel(self, source_idx):
        """Remove all information regarding labels for `source_id`."""
        # Compute idx to go back to
        
        # Don't touch history_pairs or first_scores, all_scores or has_results
    
        if source_idx in self.first_is_match:
            del self.first_is_match[source_idx]
            del self.any_is_match[source_idx]
        else:
            assert source_idx not in self.any_is_match

        # TODO: Check whether or not to re-compute scores 


    def compute_metrics(self, t_p=0.965, t_r=0.3, source_indices=None):
        """ 
        Compute the optimal threshold and the associated metrics 
        
        Parameters
        ----------
        t_p: float between 0 and 1
            target_precision
        t_r: float between 0 and 1
            target_recall
        source_indices: list of indices or None
            The indices of the source on which to restrict the computation of
            the metrics. None indicates no restriction
            
        Returns
        -------
        thresh: float
            This refers to the threshold on the Elasticsearch score for this 
            query. If the ES score of the best hit is above this threshold, it 
            can be considered to be a match; if not, it can be assumed that the 
            pair is not a match (or at least uncertain).
                    
            The threshold is computed so as to maximize the estimated query 
            score (based on precision and recall).
        precision: float between 0 an 1
            The precision corresponding to that computed using the optimal 
            threshold.
        recall: float between 0 an 1
            The recall corresponding to that computed using the optimal 
            threshold.
        score: float between 0 an 1
            The optimal score (that yields the optimal threshold).
        """
        if source_indices is None:
            source_indices = list(self.any_is_match.keys())
        # print('Len source_indices: {}'.format(len(source_indices)))
        
        summaries = [{'score': self.first_scores[source_idx], 
                      'has_results': self.has_results[source_idx],
                      'first_is_match': self.first_is_match[source_idx], 
                      'any_is_match': self.any_is_match[source_idx]} \
                      for source_idx in self.any_is_match.keys() \
                      if source_idx in source_indices]
        
        # print('Using {} summaries to compute score'.format(len(summaries)))
        
        # Filter out relevent summaries only (line not forgotten)
        summaries = [summary for summary in summaries if summary['first_is_match'] is not None]
        # print('Len summaries: {}'.format(len(summaries)))
        
        #
        num_summaries = len(summaries)
        num_has_results = sum(x['has_results'] for x in summaries)
    
        # Sort summaries by score / NB: 0 score deals with empty hits
        summaries = sorted(summaries, key=lambda x: x['score'], reverse=True)
        
        # score_vect = np.array([x['_score_first'] for x in sorted_summaries])
        is_first_vect = np.array([bool(x['first_is_match']) for x in summaries])
        has_res_vect = np.array([x['has_results'] for x in summaries]) # for assertion below only
        
        # If no matches are present
        # TODO: eliminates things from start ?
        if sum(is_first_vect) == 0:
            self.thresh = 10**3
            self.precision = 0
            self.recall = 0
            self.score = 0 
            return self.thresh, self.precision, self.recall, self.score
     
        assert is_first_vect[~has_res_vect].sum() == 0 # Check that if there are no results, there are no first
        rolling_precision = is_first_vect.cumsum() / np.minimum(np.arange(num_summaries) + 1, num_has_results)
        rolling_recall = is_first_vect.cumsum() / num_summaries
        
        # Compute custom score
        _f_precision = lambda x: (2*max(x - t_p, 0) + min(t_p*(x/t_p)**4, t_p))**1.5 / 2
        _f_recall = lambda x: max(x - t_r, 0) + min(t_r*(x/t_r)**4, t_r)
        a = np.fromiter((_f_precision(xi) for xi in rolling_precision), rolling_precision.dtype)
        b = np.fromiter((_f_recall(xi) for xi in rolling_recall), rolling_recall.dtype)
        rolling_score = a*b
    
        # Find best index for threshold
        MIN_OBSERVATIONS = 4 # minimal number of values on which to comupute threshold etc...
        idx = max(num_summaries - rolling_score[::-1].argmax() - 1, min(MIN_OBSERVATIONS, num_summaries-1))
        
        # If the best score is obtained using all results (no ES score 
        # thresholding), then set threshold to ~0 
        if idx == len(summaries) - 1:
            thresh = 0.0001
        else:
            thresh = summaries[idx]['score']
        
        precision = rolling_precision[idx]
        recall = rolling_recall[idx]
        
        # Initial version
        score = rolling_score[idx]
        
        # Version taking into account the number of matches...
        
        # Compute 
        inclusion_ratio = sum(x['any_is_match'] for x in summaries) / len(summaries)
        
        self.thresh = thresh
        self.precision = precision
        self.recall = recall
        self.score = score
        
        #print(self.precision, self.recall)
        
        return self.thresh, self.precision, self.recall, self.score


    def multiply_by_boost(self, boost_multiplier=2):
        """Return multiple variations of the current `LabellerQueryTemplate` 
        with different boost levels.
        
        This will create multiple versions of the current `LabellerQueryTemplate`
        by taking each `SingleQueryTemplate` element and generating two new 
        examples by multiplying its boost level by `boost_multiplier` (and then
        normalizing to keep a constant sum of boost levels). The output is 
        therefore of length the 2*(number of single query template in `self`).
        
        Parameters
        ----------
        boost_multiplier: float
            The value by which to multiply boost levels. Making it too high will
            will create unballanced queries, but making it too close to 1 will
            change very little from the original query.

        Returns
        -------
        new_query_templates: list of query templates        
        """
        new_query_templates = []
        
        new_query = copy.deepcopy(self) 
        new_query_templates.append(new_query) # Keep original query
        
        og_boost_total = sum(x.boost for x in self.musts + self.shoulds)
        for level in ['shoulds', 'musts']:
            for i in range(len(self.__dict__[level])):
                new_query = copy.deepcopy(self)
                new_boost_total = og_boost_total + new_query.__dict__[level][i].boost
                new_query.__dict__[level][i].boost *= 2
                
                # Normalize total boost back to original level
                for level_2 in ['shoulds', 'musts']:
                    for j in range(len(self.__dict__[level_2])):
                        new_query.__dict__[level_2][j].boost *= og_boost_total/new_boost_total
                new_query_templates.append(new_query)
                
        return new_query_templates
    
    def multiply_by_core(self, core_queries, bool_levels):
        """
        Return multiple variations of the current `LabellerQueryTemplate` 
        with additional `CoreScorerQueryTemplate`.
        
        Parameters
        ----------
        core_queries: list of `CoreScorerQueryTemplate` (or `SingleQueryTemplate`)
            Queries to add to add to the current instance.
        bool_levels: list of 1 or two elements of ['must', 'should']
            The bool levels to combine with the new core queries
            
        Returns
        -------
        new_query_templates: list of query templates 
        """
        
        new_query_templates = []
        
        new_query = copy.deepcopy(self)
        new_query_templates.append(new_query) # Keep original query
        
        # Multiply by core queries and bool levels
        for core_query in core_queries:
            for level in bool_levels:
                if level == 'must':
                    if all(core_query.core != q.core for q in self.musts):
                        new_query = copy.deepcopy(self)
                        new_query.add_must(core_query)
                        new_query_templates.append(new_query)
                elif level == 'should':
                    if all(core_query.core != q.core for q in self.musts):
                        new_query = copy.deepcopy(self)
                        new_query.add_should(core_query)
                        new_query_templates.append(new_query)   
                else:
                    raise ValueError('Invalid level: {0}; should be must or should')
        return new_query_templates        
        
    
    def new_template_restricted(self, cores_to_remove, bool_levels_to_keep):
        """ Return the a new LabellerQueryTemplate based on the current instance, 
        but for which none of the single query templates are included
        in `cores_to_remove`.
        
        This restriction can be used to limit unecessary use of resources
        by cutting out unefficient elements of a query.
        
        Parameters
        ----------
        cores_to_remove: list of `CoreScorerQueryTemplate` (or `SingleQueryTemplate`)
            The query templates to remove for the current instance.
        bool_levels_to_keep: list of 1 or two elements of ['must', 'should']
            The levels bool levels to include in the new query template.
        
        Returns
        -------
        LabellerQueryTemplate or None
            The result of the restriciton if successfull or `None` if all 
            single queries were filetered out.
            
        """
        new_single_query_template_tuples = tuple(s_q_t._as_tuple() for s_q_t in \
                    self.musts + self.shoulds if (s_q_t.core not in cores_to_remove) \
                                                  and (s_q_t.bool_lvl in bool_levels_to_keep))
        
        if new_single_query_template_tuples:
            return LabellerQueryTemplate(new_single_query_template_tuples)
        else:
            return None

class CoreScorerQueryTemplate(SingleQueryTemplate):
    """Class to evaluate the effeciency of a SingleQueryTemplate in the context
    of labelling (with `Labeller`).
    """
    
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)    
        
        self.source_lens = []
        self.ref_lens = []
        self.intersect_lens = []
        self.is_match = []
    
    def _sanity_check(self):
        assert len(self.source_lens) \
                == len(self.ref_lens) \
                == len(self.intersect_lens)
    
    @staticmethod
    def _analyze(es, index_name, analyzer, text):
        """Analyze text using Elasticsearch"""
        ic = client.IndicesClient(es)
        try:
            if analyzer:
                return ic.analyze(index_name, body={'text': text, 'analyzer': analyzer})
            else:
                return ic.analyze(index_name, body={'text': text})
        except RequestError:
            logging.warning('No analyzer {0} for index {1}'.format(analyzer, index_name))
            return {'tokens': []}
        
    def analyze_pair_items(self, es, index_name, source_item, ref_item):
        """Count tokens (from `_analyze`) for source, ref and the intersection 
        of both.
        """  
        analyzer = self.analyzer_suffix.strip('.')
        
        text_source = ' '.join(source_item[col] for col in self.source_col if source_item[col] is not None)
        tokens_source = {x['token'] for x in self._analyze(es, index_name, analyzer, text_source)['tokens']}

        text_ref = ' '.join(ref_item[col] for col in self.ref_col  if ref_item[col] is not None)
        tokens_ref = {x['token'] for x in self._analyze(es, index_name, analyzer, text_ref)['tokens']}

        return len(tokens_source), len(tokens_ref), len(tokens_source & tokens_ref)
    
    def add_labelled_pair_items(self, es, index_name, 
                                source_item, ref_item, is_match=True):
        """Update the instance with a user labelled pair.
        
        Parameters
        ----------
        es: Instance of `Elasticsearch`
            The elasticsearch connection to use for analysis
        index_name: str
            The Elasticsearch index used to analyze items
        source_item: `pandas.Series` or `dict` shaped as {column: value, ...}
            The source item
        ref_item: `pandas.Series` or `dict` shaped as {column: value, ...}
            The reference item labelled as match with `source_item`
        is_match: bool
            Whether or not the pair that was passed is a match 
        """
        l_s, l_r, l_i = self.analyze_pair_items(es, index_name, source_item, ref_item)
        
        self.source_lens.append(l_s)
        self.ref_lens.append(l_r)
        self.intersect_lens.append(l_i)
        self.is_match.append(is_match)
        
        self.update_score()
    
    def unlabel(self):
        """Remove the last pair added and recompute score."""
        self.source_lens.pop()
        self.ref_lens.pop()
        self.intersect_lens.pop()
        self.is_match.pop()
        
        self.update_score()        
        
    
    def update_score(self):
        """Update the score for this query template."""
        
        # Score is mean number of intersections on match
        match_intersect_lens = [x for x, y in zip(self.intersect_lens, self.is_match) if y]
        
        # If there are no values to score on set score of 0
        if not match_intersect_lens:
            self.score = 0
            self.yes_or_none_score = 0
            return
        
        self.score = sum(x > 0 for x in match_intersect_lens) / len(match_intersect_lens)
        
        # Yes or None on match score. Proportion of matches that have at least
        # one token in common or no token in common.
        source_inter_lens = [x for x, y in zip(self.source_lens, self.is_match) if y]
        ref_inter_lens = [x for x, y in zip(self.ref_lens, self.is_match) if y]
        inter_lens = [x for x, y in zip(self.intersect_lens, self.is_match) if y]
        self.yes_or_none_score = sum((i>0) or (s==0) or (r==0)  \
                 for (i, s, r) in zip(inter_lens, source_inter_lens, ref_inter_lens)) \
                 / len(inter_lens)
        
        # NB: no need to use yes or None if already using the core as must
        
    def to_dict(self):
        """Returns a dict representation of the instance."""
        dict_ = super().to_dict()
        
        if hasattr(self, 'score'):
            dict_['score'] = self.score    
            
        dict_['source_lens'] = self.source_lens
        dict_['ref_lens'] = self.ref_lens
        dict_['intersect_lens'] = self.intersect_lens
        dict_['is_match'] = self.is_match
        return dict_
    
    @classmethod
    def from_dict(cls, dict_):
        """Returns an instance of the class using a representation generated 
        by to_dict.
        """
        csqt = super(CoreScorerQueryTemplate, cls).from_dict(dict_)
        
        if 'score' in dict_:
            csqt.score = dict_['score']
            
        csqt.source_lens = dict_['source_lens']
        csqt.ref_lens = dict_['ref_lens']
        csqt.intersect_lens = dict_['intersect_lens']
        csqt.is_match = dict_['is_match']
        return csqt
    

class BasicLabeller():
    """Object used for interactive learning of optimal queries to match two
    tables.
    
    The instance of `Labeller` is used to link a source file (`pandas.DataFrame`)
    with a file indexed in Elasticsearch (see docs for help). It proposes
    matches between source and referential which the user has to label. It uses
    these label to score different query templates (`LabellerQueryTemplates`).
    
    The general process goes as follows:
        0. Initiate queries/ metrics/ history
        1. Read first row
        
        2. Perform queries / update history of hits
        3. Generate pairs to propose (based on sorted queries)
        4. Until row is over: User inputs label
        5. Update metrics and history 
        6. Sort queries
        7. Generate new row and back to 2
        
    Examples
    --------
    >>> labeller = Labeller()
    >>> for x in range(100):
            to_display = labeller.to_emit()
            labeller.update('y') # Or 'n' (no) or 'p' (previous) or 'u' (unkown)      
            
    Attributes
    ----------
    self.current_queries: list of `LabellerQueryTemplate`
        A list of query templates that is being evaluated to determin which 
        is the best query template to use for the current matching problem.
    self.single_core_queries: list of `CoreScorerQueryTemplate`
        A list of single queries that is being evaluated to help filtering 
        and generating pertinant queries for self.current_queries.
    self.must_filters: `dict` shaped as {column: list_of_words, ...}
        Positive filters on fields of the referential (to force results to 
        include certain words in certain fields).
    self.must_not_filters: `dict` shaped as {column: list_of_words, ...}
        Negative filters on fields of the referential (to NOT return results 
        that include certain words).
    """
    NUM_SEARCH_RESULTS = 3
    
    MAX_NUM_SAMPLES = 100
    
    VALID_ANSWERS = {'yes': 'y',
                 'y': 'y',
                 '1': 'y',
                 
                 'no': 'n',
                 'n': 'n',
                 '0': 'n',
                 
                    #                 'uncertain': 'u',
                    #                 'u': 'u',
                 
                 'forget_row': 'f',
                 'f': 'f',
                 'uncertain': 'f',
                 'u': 'f',
                 
                 'previous': 'p',
                 'p': 'p'         
                 }
    
    MAX_NUM_LEVELS = 3 # Number of match clauses
    MIN_NUM_QUERIES = 3 # Minimum number of queries to try out
    MAX_NUM_QUERIES_FOR_LINKING = 6 # Number of queries for export
    BOOL_LEVELS = {'.integers': ['must', 'should'], # Analyzers with "should"
                   '.city': ['must', 'should']}
    BOOST_LEVELS = [1]
    
    # Target precision and target recall
    TARGET_PRECISION = 0.95
    TARGET_RECALL = 0.3    
    
    @staticmethod
    def _dedupe_source(source, match_cols):
        """Dedupe source on columns used for matching to avoid the user having 
        to label the same elements multiple times.
        """
         # Dedupe source on matching columns
        source_cols_for_match = set()
    
        for match in match_cols:
            if isinstance(match['source'], str):
                source_cols_for_match.add(match['source'])
            else:
                source_cols_for_match.update(match['source'])

        source_cols_for_match = list(source_cols_for_match)       
        
        smaller_source = source.drop_duplicates(subset=source_cols_for_match)
        
        return smaller_source
    
    def __init__(self, es, source, ref_index_name, 
                 match_cols, columns_to_index,
                 must={}, must_not={}, next_row=True):
        """        
        Parameters
        ---------- 
        es: Instance of `Elasticsearch`
            The elasticsearch connection to use for analysis.
        source: `pandas.DataFrame`
            Source (dirty data) containing input data to match with reference.
        ref_index_name: str
            Name of the Elasticsearch index containg the reference data to
            match with source.
        match_cols: dict
            Indication of columns to try to match:
            Ex: [{'source': ['city'], 'ref': ['CITY_ADD']},
                 {'source': ['first_name', 'last_name'], 'ref': ['FULL_NAME']}]
        columns_to_index: dict
            Dictionnary linking each column of the referential to the analyzers 
            used for indexing.
            Ex: {
                'CITY_ADD': {},
                'FULL_NAME': {'french', 'integers', 'n_grams', 'city'}
                }
        must: dict of shape {column: list_of_strings, ...}
            Filters retusults on values in columns of referential (must have match).
        must_not: dict of shape {column: list_of_strings, ...}
            Exclude these words from results in referential (cannot have match)
        """
        
        self.source = self._dedupe_source(source, match_cols)
        self.ref_index_name = ref_index_name
        
        def _unlist_match(x):
            def _unlist(y):
                if isinstance(y, list):
                    return tuple(y)
                else:
                    return y
            return {'source': _unlist(x['source']), 'ref': _unlist(x['ref'])}
            
        match_cols = [_unlist_match(match) for match in match_cols]
           
        self.match_cols = match_cols
        
        columns_to_index = {key: set(values) for key, values in columns_to_index.items()}
        self.columns_to_index = columns_to_index
        self.es = es
        
        self.labelled_pairs = [] # Flat list of labelled pairs
        self.labels = {} # Flat list of labels
        self.num_rows_labelled = [] # Flat list: at given label, how many were labelled. NB: starts at 0, becomes 1 at first yes/forgotten, or when next_row
        self.num_positive_rows_labelled = [] # Flat list: at given label, how_many were matches
        self.labelled_pairs_match = [] # For each row, the resulting match: (A, B) / no-match: (A, None) or forgotten: None
         
        self.must_filters = must
        self.must_not_filters = must_not
           
        self._init_queries(match_cols, columns_to_index) # creates self.current_queries
        self.MIN_NUM_QUERIES = min(self.MIN_NUM_QUERIES, len(self.current_queries)) 
        self._init_core_queries(match_cols, columns_to_index) # creates self.single_core_queries
        #self._init_history() # creates self.history
        
        self._init_source_gen() # creates self.source_gen
        
        self.current_source_idx = None
        self.current_ref_idx = None
        
        self.current_source_item = None
        self.current_ref_item = None
        
        self.current_es_score = None
        
        #
        self.estimated_precision = 0
        self.estimated_recall = 0
        self.num_queries_sorted = 1 # Number of queries sorted by _iterative_metrics_and_sort
        
        self.status = 'ACTIVE' # 'ACTIVE', 'NO_ITEMS_TO_LABEL', 'NO_QUERIES', 'NO_MATCHES'

        if next_row:
            self._next_row()

        self._sanity_check()
        

    def _sanity_check(self):
        """Make sure you are not crazy. Run this after updates in dev."""
        return
    
        # Labels
        if self.status == 'ACTIVE':
            assert len(self.labelled_pairs) \
                        == len(self.labels)
            
            assert len(self.num_rows_labelled) \
                    == len(self.num_positive_rows_labelled)

        assert all(isinstance(x, tuple) for x in self.labelled_pairs)
        
        # Queries
        assert len(self.current_queries) == len(set(self.current_queries))
        
        for query in self.current_queries:
            query._sanity_check()
            
        for query in self.single_core_queries:
            query._sanity_check()
        

    def to_dict(self):
        """Returns a dict representation of the instance."""
        # source and ref_index_name, es, source_gen, ref_gen, and all current 
        # except for queries are not included
        
        dict_ = dict()
        
        dict_['match_cols'] = self.match_cols
        dict_['columns_to_index'] = self. columns_to_index
        
        dict_['labelled_pairs'] = self.labelled_pairs
        dict_['labels'] = _key_val_er(self.labels)
        dict_['num_rows_labelled'] = self.num_rows_labelled
        dict_['num_positive_rows_labelled'] = self.num_positive_rows_labelled
        dict_['labelled_pairs_match'] = self.labelled_pairs_match

        dict_['must_filters'] = self.must_filters
        dict_['must_not_filters'] = self.must_not_filters
        
        dict_['current_queries'] = [query.to_dict() for query in self.current_queries]   
        dict_['single_core_queries'] = [query.to_dict() for query in self.single_core_queries]   
        
        if self.current_query is None:
            dict_['current_query'] = None
        else:
            dict_['current_query'] = self.current_query.to_dict()
        dict_['current_query_ranking'] = self.current_query_ranking
        
        dict_['current_source_idx'] = self.current_source_idx
        dict_['current_ref_idx'] = self.current_ref_idx
        if isinstance(self.current_source_item, dict):
            dict_['current_source_item'] = self.current_source_item
        else:
            dict_['current_source_item'] = self.current_source_item.to_dict() # origin
        dict_['current_ref_item'] = self.current_ref_item
        dict_['current_es_score'] = self.current_es_score
        
        dict_['estimated_precision'] = self.estimated_precision
        dict_['estimated_recall'] = self.estimated_recall
        dict_['num_queries_sorted'] = self.num_queries_sorted
        
        dict_['ref_id_to_data'] = self.ref_id_to_data
        
        dict_['status'] = self.status
        
        dict_['TARGET_PRECISION'] = self.TARGET_PRECISION
        dict_['TARGET_RECALL'] = self.TARGET_RECALL
        
        # self._init_source_gen() # creates self.source_gen
        return dict_
        
    
    @classmethod
    def from_dict(cls, es, source, ref_index_name, dict_):
        """Returns an instance of the class using a representation generated 
        by to_dict.
        
        Parameters
        ----------
        es: Instance of `Elasticsearch`
            See doc for `__init__`. 
        source: `pandas.DataFrame`
            See doc for `__init__`. 
        ref_index_name: str
            See doc for `__init__`. 
        dict_: dict        
            The dictionnary to load the `Labeller` instance from (result of
            `to_dict`).
        """        
        labeller = cls(es, source, ref_index_name, dict_['match_cols'], 
                                                dict_['columns_to_index'],
                                                dict_['must_filters'], 
                                                dict_['must_not_filters'],
                                                next_row=False)
        
        # Load from dict
        labeller.labelled_pairs = [tuple(x) for x in dict_['labelled_pairs']]
        labeller.labels = _un_key_val_er(dict_['labels'] )
        labeller.num_rows_labelled = dict_['num_rows_labelled'] 
        labeller.num_positive_rows_labelled = dict_['num_positive_rows_labelled'] 
        labeller.labelled_pairs_match = [tuple(x) for x in dict_['labelled_pairs_match']]         
        
        labeller.current_queries = [LabellerQueryTemplate.from_dict(x) for x in dict_['current_queries']]
        labeller.single_core_queries = [CoreScorerQueryTemplate.from_dict(x) for x in dict_['single_core_queries']]
        
        labeller._init_source_gen() # creates self.source_gen
        
        if dict_['current_query'] is None:
            labeller.current_query = None
        else:
            labeller.current_query = LabellerQueryTemplate.from_dict(dict_['current_query'])
        labeller.current_query_ranking = dict_['current_query_ranking']
        
        labeller.current_source_idx = dict_['current_source_idx']
        labeller.current_ref_idx = dict_['current_ref_idx']
        
        labeller.current_source_item = dict_['current_source_item']
        labeller.current_ref_item = dict_['current_ref_item']
        
        labeller.current_es_score = dict_['current_es_score']
        
        labeller.estimated_precision = dict_['estimated_precision']
        labeller.estimated_recall = dict_['estimated_recall']
        labeller.num_queries_sorted = dict_['num_queries_sorted']

        labeller.ref_id_to_data = dict_['ref_id_to_data']

        labeller.status = dict_['status']
        
        labeller.TARGET_PRECISION = dict_.get('TARGET_PRECISION', 0.95)
        labeller.TARGET_RECALL = dict_.get('TARGET_RECALL', 0.3)

        labeller._sanity_check()
        
        labeller._init_source_gen()
        labeller._init_ref_gen() 

        return labeller

    def to_json(self, file_path):
        """Use `to_dict` to write the current Labeller as JSON file.
        """        
        
        dict_ = self.to_dict()
        encoder = MyEncoder()
        
        try:
            with open(file_path, 'w') as w:
                w.write(encoder.encode(dict_))
        except:
            for key in dict_.keys():
                with open(file_path, 'w') as w:
                    w.write(encoder.encode(dict_[key]))         

    @classmethod
    def from_json(cls, file_path, es, source, ref_index_name):
        """Load a `Labeller` instance from a JSON file writen by `to_json`.
        
        Parameters
        ----------
        es: Instance of `Elasticsearch`
            See doc for `__init__`. 
        source: `pandas.DataFrame`
            See doc for `__init__`. 
        ref_index_name: str
            See doc for `__init__`. 
        dict_: dict        
            The dictionnary to load the `Labeller` instance from (result
            of `to_json`).
        """
        with open(file_path) as f:
            dict_ = json.load(f)
        labeller = cls.from_dict(es, source, ref_index_name, dict_)
        return labeller    
    
    def _fetch_source_item(self, source_idx):
        """Fetch source item from pandas table in memory."""
        return self.source.loc[source_idx, :]
    
    def _fetch_ref_item(self, ref_idx):
        """Fetch ref item from Elasticsearch database."""
        # TODO: look into batching this
        return self.es.get(self.ref_index_name, 'structure', ref_idx)['_source'] 

    def _init_queries(self, match_cols, columns_to_index):
        """Generate initial query templates to be assigned to `current_queries`.
        """
        all_query_template_tuples = _gen_all_query_template_tuples(match_cols, 
                                                           columns_to_index, 
                                                           self.BOOL_LEVELS, 
                                                           self.BOOST_LEVELS, 
                                                           self.MAX_NUM_LEVELS)        
        self.current_queries = [LabellerQueryTemplate(q_t_t) for q_t_t in all_query_template_tuples]            
        self._prune_analyzers(remove_duplicates=True)
    
    def _init_core_queries(self, match_cols, columns_to_index):
        """Generate initial core query templates. to assign to `single_core_queries`."""
        all_single_query_templates_tuples = _gen_all_single_query_template_tuples(
                                                    match_cols, 
                                                    columns_to_index, 
                                                    {}, # defaults to must
                                                    [1])
        self.single_core_queries = [CoreScorerQueryTemplate(*q_t_t) for q_t_t \
                                    in all_single_query_templates_tuples]

    
    def _init_source_gen(self):
        """Generator of rows of source to label."""
        def temp():
            sources_done = [x[0] for x in self.labelled_pairs_match if x is not None] # TODO: forgotten can be re-labelled
            for idx in random.sample(list(self.source.index), 
                                     min(len(self.source), self.MAX_NUM_SAMPLES)):
                if idx not in sources_done:
                    item = self._fetch_source_item(idx) 
                    
                    self.current_source_idx = idx # Redundency with yield 
                    self.current_source_item = item
                                       
                    yield (idx, item)
        self.source_gen = temp()
    
    def _fetch_results_for_row(self):
        """Fetch data for all current queries for the current source item, 
        and call `add_results`.
        """
        if self.current_source_idx not in self.current_queries[0].history_pairs:
            results = self.pruned_bulk_search(self.current_queries, 
                            self.current_source_item, self.NUM_SEARCH_RESULTS)
            self.add_results(results)


    def _is_exact_match(self, source_item, ref_item):
        """Check if match is considered exact between two items on columns used
        for matching.
        """
        
        for match in self.match_cols:
            s_col = match['source']
            if isinstance(s_col, str):
                s_col = [s_col]
                
            r_col = match['ref']
            if isinstance(r_col, str):
                r_col = [r_col]
                
            s_string = ' '.join(source_item[col] for col in s_col)
            
            if not any(s_string == ref_item[col] for col in r_col):
                return False
        return True
    
    @time_in
    def _get_ref_item(self, ref_idx):
        '''Get the reference item associated to the index by searching in 
        memory first and then Elasticsearch.'''
        # If the data associated to id is in memory
        if ref_idx in self.ref_id_to_data:
            item = self.ref_id_to_data[ref_idx]['_source']
        # If the data is not in memory, fetch by ID
        else:
            item = self._fetch_ref_item(ref_idx)
        return item

    def _get_source_item(self, source_idx):
        '''Get the reference item associated to the index.'''
        return self._fetch_source_item(source_idx)  
    
    @time_in
    def _ref_rows_for_current_source_row(self, max_num_items, max_num_queries):
        ''' /!\ This function overlaps with _init_ref_gen and could be used in a 
        future simpler version of the project in combination with label_pair
        '''
        MIN_ES_SCORE_TO_THRESH = 2
        MAX_NUM_PROPOSALS_PER_QUERY = 4

        # Fetch data for current row
        self._fetch_results_for_row()
        
        ref_rows = []
        num_items = 0
        try:
            for i, query in enumerate(self.current_queries[:max_num_queries]):
    #            self.current_query_ranking = i
    #            self.current_query = query # TODO: for display only
                ref_rows_query = []
                
                for pair in query.history_pairs[self.current_source_idx][:MAX_NUM_PROPOSALS_PER_QUERY]: 
                    assert pair[0] == self.current_source_idx
    
                    # Check that source does not have match (or No match)
                    # Check that pair was not already labelled          
                    if pair[1] not in [item['_id'] for items in ref_rows for item in items]:
                    
                        item = self._get_ref_item(pair[1])
                        es_score = query._get_score_for_pair(pair[0], pair[1])
                        
                        # Yield only if probable enough
                        if query.thresh is not None:
                            if es_score >= (query.thresh / MIN_ES_SCORE_TO_THRESH):
                                ref_rows_query.append({'_id': pair[1], '_source': item, '_score': es_score, '_query_thresh': query.thresh})
                        else:
                            ref_rows_query.append({'_id': pair[1], '_source': item, '_score': es_score, '_query_thresh': None})
                        num_items += 1
                        
                        if num_items >= max_num_items:
                            raise StopIteration
                ref_rows.append(ref_rows_query)
        except StopIteration:
            pass
        
        print(sum(1 for y in ref_rows for x in y))
        return ref_rows
                                
                                
    def _init_ref_gen(self):
        """Initialize `ref_gen`.
        
        Initialize a generator of pairs to label for the source element 
        currently being labelled (`current_source_idx`).
        """
        
        MIN_ES_SCORE_TO_THRESH = 2
        MAX_NUM_PROPOSALS_PER_QUERY = 4

        # Fetch data for current row
        self._fetch_results_for_row()
        
        def temp():
            for i, query in enumerate(self.current_queries):
                self.current_query_ranking = i
                self.current_query = query # TODO: for display only
                         
                for pair in query.history_pairs[self.current_source_idx][:MAX_NUM_PROPOSALS_PER_QUERY]: 
                    
                    assert pair[0] == self.current_source_idx

                    # Check that source does not have match (or No match)
                    if pair[0] not in [x[0] for x in self.labelled_pairs_match if x is not None]:
                        # Check that pair was not already labelled                        
                        if pair not in self.labelled_pairs:
                            
                            item = self._get_ref_item(pair[1])
                            es_score = query._get_score_for_pair(pair[0], pair[1])
                            
                            # Yield only if probable enough
                            if query.thresh is not None:
                                if es_score >= (query.thresh / MIN_ES_SCORE_TO_THRESH):
                                    yield pair[1], item, es_score
                            else:
                                yield pair[1], item, es_score
                                
                        # TODO: check that source idx is same as in source_gen
        self.ref_gen = temp()        

    def _next_row(self):
        """Change labeller state to following row of source.
        
        Update the current source and reference row currently being labelled. 
        This assumes that `source_gen` was initialized.
        """
        NUM_ROW_TRIES = 20
   
        if sum(x[1]=="__NO_RESULT" for x in self.labelled_pairs_match[-NUM_ROW_TRIES:]) == NUM_ROW_TRIES: #check if number of no results > NUM_ROW_TRIES
            self.status = 'NO_MATCHES'
            logging.warning('Could not find any resut in {0} consecutive rows'.format(NUM_ROW_TRIES))
            return
     
        try:
            self.current_source_idx, self.current_source_item = next(self.source_gen)
        except:
            self.status = 'NO_ITEMS_TO_LABEL'
        else:
            self._init_ref_gen()
            try: 
                (self.current_ref_idx, self.current_ref_item, self.current_es_score) = next(self.ref_gen)
            except StopIteration:
                print(self.current_source_item)
                print('WARNING: no results found for the row above; skipping')
                self.update_pair(self.current_source_idx, None, "f") # This will call NEXT ROW (careful not to loop !)
        
        


    @time_in
    def _bulk_search(self, queries_to_perform, row, num_results):
        """Perform bulk searches using Elasticsearch (wrapper around 
        helpers._bulk_search
        
        Parameters
        ----------
        queries_to_perform: list of `CompoundQueryTemplate` 
            The queries templates to be used for search
        row: `pandas.Series` or `dict` like {column: value, ...}
            Elements to search for in query
        num_results: int
            Maximum number of elements to return using Elasticsearch query
        """
        # TODO: use self.current_queries instead ?
        
        # Transform
        queries_to_perform_tuple = [x._as_tuple() for x in queries_to_perform]
        
        search_templates, full_responses = _bulk_search(self.es, 
                                             self.ref_index_name, 
                                             queries_to_perform_tuple, 
                                             [row],
                                             self.must_filters, 
                                             self.must_not_filters, 
                                             num_results)
        
        assert [x[1][0] for x in search_templates] == queries_to_perform_tuple
        
        new_full_responses = [full_responses[i]['hits']['hits'] for (i, _) in search_templates]
        return new_full_responses
        
    @print_name
    def pruned_bulk_search(self, queries_to_perform, row, num_results):
        """ Performs a smart bulk request to search for a single row with 
        with multiple analyzer combinations; optimized for a large number of 
        queries.
        
        This function is a wrapper around bulk_search that organizes the search
        by branches based on common query cores. It will not search for templates
        if restrictions of these templates already did not return any results.
        
        Parameters
        ----------
        queries_to_perform: list of instances of `CompoundQueryTemplate`
        row: `pandas.Series` or `dict` like {column: value, ...}
            Elements to search for in query.
        num_results: int
            Maximum number of elements to return using the Elasticsearch query.
        """
        return _pruned_bulk_search(self.es, self.ref_index_name, 
                                   queries_to_perform, 
                                   row, 
                                   self.must_filters, 
                                   self.must_not_filters, 
                                   num_results)

    @time_in
    def _compute_metrics(self, source_indices=None, query_ids=None):
        """Compute metrics for each individual query.
        
        Parameters
        ----------
        source_indices: list of indices or None
            The indices of the source on which to restrict the computation of
            the metrics. None indicates no restriction.
        query_ids: list of query ids
            The indices of the queries for which to re-compute metrics. Use 
            None to compute on all
        """
        
        if query_ids is None:
            for query in self.current_queries:
                query.compute_metrics(self.TARGET_PRECISION, self.TARGET_RECALL, 
                                      source_indices)
        else:
            print('Compute metrics (on select queries) computing on {} queries'.format(sum(1 for q in self.current_queries if (q.id_ in query_ids))))
            for query in [q for q in self.current_queries if (q.id_ in query_ids)]:
                query.compute_metrics(self.TARGET_PRECISION, self.TARGET_RECALL, 
                                      source_indices)            
            
    def _metrics_and_sort(self):
        METHOD = 'iterative'
        
        if METHOD == 'default':
            self._default_metrics_and_sort()
        
        elif METHOD == 'iterative':
            self._iterative_metrics_and_sort()
            
        
    @time_in
    def _default_metrics_and_sort(self):
        self._sorta_sort_queries()
        self._compute_metrics()
        self._sort_queries()       
        
        self.estimated_precision = self.current_queries[0].precision
        self.estimated_recall = self.current_queries[0].recall
        
        self.num_queries_sorted = 1 #max(len(self.current_queries))
    
    @time_in
    def _iterative_metrics_and_sort(self):
        """Iteratavely sort queries, recomputing scores based on the remaining
        non-matched examples.
        
        # TODO: try multiple orderes if there are equal scores
        """
        self._sorta_sort_queries()
        self._compute_metrics()
        self._sort_queries()
        
        # Max number of queries to cover (at least one)
        max_iterations = min(max((self._nprl()-4) // 2, 1), self.MAX_NUM_QUERIES_FOR_LINKING)
        
        sources_unmatched = set(self.current_queries[0].first_is_match.keys())
        sources_matched = set()
        num_valid_matches = 0
        ordered_queries = []
        unordered_queries = [q for q in self.current_queries]


        SORT_ON = 'score'
        for i in range(0, min(max_iterations, len(self.current_queries))):
            print('Checking to add query {}'.format(i))
            if not sources_unmatched:
                break
            if (len(sources_unmatched) <= 3) and i > 0:
                break
            self._compute_metrics(query_ids=[q.id_ for q in unordered_queries], 
                                  source_indices=sources_unmatched)
            
            # Get best query with the subset of sources being considered
            query = unordered_queries[np.argmax([getattr(q, SORT_ON) for q in unordered_queries])]
            
            # Add query only if it improves the global score 
            # TODO: check if we should do that
            if bool(ordered_queries) and (query.score < ordered_queries[-1].score):
                break
            
            unordered_queries = [q for q in unordered_queries if q != query] # TODO: check efficiency
            ordered_queries.append(query)
            print('Chose query with precision {} and recall {}'.format(query.precision, query.recall))
            
            # Evaluate each un-matched source with the current query being considered
            for source_id, es_score in query.first_scores.items():
                if source_id in sources_unmatched:
                    if es_score >= query.thresh:
                        sources_matched.add(source_id)
                        sources_unmatched.remove(source_id)
                        num_valid_matches += int(query.first_is_match[source_id])
            
            # TODO: evaluate contextualized precision and recall
        if sources_matched:
            self.estimated_precision =  num_valid_matches / len(sources_matched)
            self.estimated_recall = num_valid_matches / (len(sources_matched) + len(sources_unmatched))
        else:
            self.estimated_precision = 0
            self.estimated_recall = 0
            
        print('Iterative sorting led to {} sorted'.format(len(ordered_queries)))
        # List all ordered queries
        
        # Put this instead of checking ordered queries directly for performance
        # issues. Check why __eq__ of queries is not called.
        iter_sorted_query_ids = [q.id_ for q in ordered_queries]
        unsorted_query_ids = []
        for query in self.current_queries:
            if query.id_ not in iter_sorted_query_ids:
                ordered_queries.append(query)
                unsorted_query_ids.append(query.id_)
                
        self.current_queries = ordered_queries
        self.num_queries_sorted = i + 1
        
        # TODO: check if we should really do this ? # Wht not before resorting of current_queries
        self._compute_metrics(query_ids=unsorted_query_ids)
        self._sort_queries(by="score")
        
        # NB: there  are cases where og_len != len(self.current_queries)
        # because the original self.current_queries has duplicates. We do not
        # Call set because it is too slow. # TODO: check why set is so slow
        #        try:
        #            assert og_len == len(self.current_queries)
        #        except:
        #            import pdb; pdb.set_trace()
    
    def majority_vote(self, max_num_voters, min_score=0):
        """ #TODO: Document """
        if not self.current_queries:
            return None
        
        count = defaultdict(int)
        for query in self.current_queries[:max_num_voters]:
            if query.history_pairs[self.current_source_idx]:
                if query.thresh is None:
                    thresh = 0
                else:
                    thresh = query.thresh
                # Add to count if source is seen as match (score above threshold)
                if query.first_scores[self.current_source_idx] >= thresh:
                    count[query.history_pairs[self.current_source_idx][0]] += 1
                else:
                    count['nores'] += 1
            else:
                count['nores'] += 1
            
        best_pair = sorted(list(count.items()), key=lambda x: x[1], reverse=True)[0][0]
        return best_pair
    
    @time_in
    def _sort_queries(self, by='score'):
        """Sort queries according to parameter (best first).
        
        Parameters
        ----------
        by: str ('score' or 'precision' or 'recall')
            Attribute to use to sort queries
        """
        self.current_queries = sorted(self.current_queries, 
                                      key=lambda x: getattr(x, by), reverse=True)
           
    @time_in
    def _sorta_sort_queries(self):
        """Alternate between random queries and sorted by best Elasticsearch score.
        
        Use this in burnout phase, when there are not enough data to compute 
        real precision, recall, scores...
        """
        # Only sort if there are values to sort on
        if (not self.current_queries) or (not self.current_queries[0].first_scores):
            return
        
        # Alternate between random and largest score
        queries = random.sample(self.current_queries, len(self.current_queries))
        
        a = queries[:int(len(queries)/2)]
        a = sorted(a, key=lambda x: x.first_scores[self.current_source_idx], reverse=True)
        
        #        self.current_queries = a
        
        if len(queries)%2 == 0:
            b = queries[int(len(queries)/2):]
            c = []
        else:
            b = queries[int(len(queries)/2):-1]
            c = [queries[-1]]
            
        # d_q = self._default_query()
        self.current_queries = [x for x in list(itertools.chain(*zip(a, b)))] + c

    @print_name
    def previous(self):
        """Restores the previous state regarding labelling.
        
        This removes the last label and puts "current" attributes to their 
        previous version. It removes the labels from current_queries but metrics
        are NOT re-computed. Also, addition of must_filters, must_not_filters 
        and any changes to current_queries (expansion or filtering) will NOT 
        be undone.    
        """
        # Re-place current values in todo-stack
        s_elem = (self.current_source_idx, self.current_source_item)
        r_elem = (self.current_ref_idx, self.current_ref_item, self.current_es_score)
        
        # If going to previous 
        # Re-put the current state in the generator
        
        og_num_rows_labelled = self._nrl()
        # TODO: test that previous is possible
        # Update the current state
        (self.current_source_idx, self.current_ref_idx) = self.labelled_pairs.pop()
        previous_label = self.labels[(self.current_source_idx, self.current_ref_idx)] # needed for previous on core queries
        del self.labels[(self.current_source_idx, self.current_ref_idx)]
        num_rows_labelled = self.num_rows_labelled.pop()
        self.num_positive_rows_labelled.pop()
        
        # Remove from labelled pairs match if there was a change of line
        if (self.num_rows_labelled and (num_rows_labelled > self.num_rows_labelled[-1])) \
                or ((not self.num_rows_labelled) and bool(num_rows_labelled)):
            self.labelled_pairs_match.pop()

        self.current_source_item = self._fetch_source_item(self.current_source_idx)
        if self.current_ref_idx is not None: # In the case of no results when fetching row
            self.current_ref_item = self._fetch_ref_item(self.current_ref_idx)
        
        self.current_es_score = None
        
        # Previous on all queries
        for query in self.current_queries:
            query.unlabel(self.current_source_idx)           

        # Previous on all core queries
        if previous_label == 'y':
            for core_query in self.single_core_queries:
                core_query.unlabel()       

        # If we just changed source row, re generate 
        if self._nrl() < og_num_rows_labelled:
            self.source_gen = itertools.chain([s_elem], self.source_gen)
            self._init_ref_gen()
        # Otherwise just add the current ref element to top of ref_gen pile
        else:
            self.ref_gen = itertools.chain([r_elem], self.ref_gen)
            
        # If there are no results for the current row, do previous again
        if self.current_ref_idx is None:
            self.previous()

        
    def re_train(self):
        
        # Fetch data for which data is missing.
        
        # OPTION 1: Go through add_labelled_data expand, filter
        
        # OPTION 2: Use only current queries
        
        pass
    
    
    def auto_label(self, certain_column_matches, 
                           num_rows_try=100, 
                           update_single_queries=True):
        """Automatically label pairs if they have a match on a subset of columns
        designated as keys.
        
        Label pairs that have equal values on columns indicated by 
        `certain_column_matches`. This is useful to automatically generate 
        training labels in the case where a file has data on a field that 
        can be used as merge key.
        
        Parameters
        ----------
        certain_columns_matches: dict or list of dict of shape {'source': col_source, 'ref': col_ref}
            Column pair or list of column pairs on which a match is equivalent
            to a match of the rows.
        num_rows_try: int
            The maximum number of rows of the source to try to auto-label.
        update_single_queries: bool
            Whether or not to update `single_core_queries`.
        """
        # TODO: Check that certain_column_matches are not in common with column_matches
        KEYWORD_ANALYZER = ''
        
        assert len(certain_column_matches['ref']) == 1        

        if isinstance(certain_column_matches['source'], str):
            certain_column_matches['source'] = [certain_column_matches['source']]
        
        # The exact match query to use for 
        query_to_perform = CompoundQueryTemplate((('must', 
                            certain_column_matches['source'], 
                            certain_column_matches['ref'],
                            KEYWORD_ANALYZER,
                            1),))
        
        
        self._init_source_gen()
        
        for _ in range(num_rows_try):
            try:
                self.current_source_idx, self.current_source_item = next(self.source_gen)
            except StopIteration:
                print('WARNING: No more rows for Auto-Labelling')
                break          
            
            
             # If certain_column_matches do not have values in source, don't auto-label
            if not any(bool(x) for x in [self.current_source_item[col] \
                           for col in certain_column_matches['source']]):
                continue
            
            # Search for the exact match in ref if 
            # TODO: No reason to bulk search here; regular search should work
            results = self._bulk_search([query_to_perform], self.current_source_item, 2)
            assert len(results) == 1
            
            if len(results[0]) == 0:
                continue
                
            if len(results[0]) > 1:
                raise RuntimeError('Results in auto-label got more than one'
                       ' in result where it expected at most one result '
                       '(certain_column_matches should be a unique identifier)')
            
            self.current_ref_idx = results[0][0]['_id']
            self.current_ref_item = results[0][0]['_source'] # _source is the ES field
            
            pair = (self.current_source_idx, self.current_ref_idx)
                  
            # > Update             
            self.labelled_pairs.append(pair)
            self.labels[pair] = 'y'     
            self.labelled_pairs_match.append(pair) # Normally done in add_labelled_pairs_match
            
            # Add rows_labelled_counts
            if pair[1] in ['__FORGET', '__NO_RESULT']:
                self._update_row_count(True, False)
            else:
                self._update_row_count(True, True)   


            if update_single_queries:
                for query in self.single_core_queries:
                    query.add_labelled_pair_items(self.es, self.ref_index_name, 
                                    self.current_source_item, self.current_ref_item)
        
        # TODO: separate learn of auto label ?        
        # Re-train (learn queries)
        self._re_score_history(True, learn=True)

    @print_name
    def add_labelled_pair(self, labelled_pair):
        """Update labeller and each individual query template once the labelling 
        of a row is over.
        
        Parameters
        ----------
        labelled_pair: tuple of len 2
            The labelled pair. One of:
                + (source_id, ref_id) (the corresponding ref if a match was found)
                + (source_id, '__FORGET')
                + (source_id, '__NO_RESULT')
        """
        print('Adding labelled pair', labelled_pair)
        for query in self.current_queries:
            query.add_labelled_pair(self.current_source_idx, labelled_pair)
        self.labelled_pairs_match.append(labelled_pair)

    @print_name
    def add_results(self, results):
        """Add results of search of each query to each individual query.
        
        Parameters
        ----------
        results: list of Elasticsearch results
            The results of searches of the current source by all queries in 
            `current_queries`.
        """
        
        # TODO: look into keeping more data than just this round
        self.ref_id_to_data = dict()
        
        assert len(self.current_queries) == len(results)
        for query, ref_result in zip(self.current_queries, results):
            # Get score of best hit or assign zero if there are no results
            if ref_result:
                score = ref_result[0]['_score']
                scores = [x['_score'] for x in ref_result]
            else:
                score = 0
                scores = []
            
            # Add the results to the query history
            query.add_pairs(self.current_source_idx, 
                            [(self.current_source_idx, x['_id']) for x in ref_result], 
                            score, 
                            scores)
            
            # Add items ('_source') to memory for faster access at this round
            for res in ref_result:
                #if res['_id'] not in self.ref_id_to_data:
                self.ref_id_to_data[res['_id']] = res

    @print_name
    def _update_row_count(self, at_new_row, last_is_match):
        """Update the count of number of rows labelled and number of rows with
        positive labels.
        
        Parameters
        ----------
        at_new_row: bool
            whether or not we are labelling a new row from source
        last_is_match: bool
            whether or not the last row labelled was a match
        """
        
        assert at_new_row or (not last_is_match)
        
        if self.num_rows_labelled:
            self.num_rows_labelled.append(self.num_rows_labelled[-1] + int(at_new_row))
            self.num_positive_rows_labelled.append(self.num_positive_rows_labelled[-1] + int(last_is_match))
        else:
            self.num_rows_labelled = [int(at_new_row)]
            self.num_positive_rows_labelled = [int(last_is_match)]
    
    @print_name
    def update(self, user_input):
        """Update the labeller according to the user input for the current pair
        being labelled.
        
        Parameters
        ----------
        user_input: str
            The user input corresponding to the "current" pair being labelled.
            + "y" or "1" or "yes": res_id is a match with self.idx
            + "n" or "0" or "no": res_id is not a match with self.idx
            + "f" or "forget_row": uncertain #TODO: is this no ?
            + "p" or "previous": back to previous state
        """
        
        print('At pair {0} / {1} ; user input: {2}'.format(self.current_source_idx, 
                                              self.current_ref_idx, user_input))
        
        # Interpret answers
        use_previous = self.VALID_ANSWERS[user_input] == 'p'
                   
        if use_previous:
            self.previous()
            return

        # Update with the pair for which the user sent an answer
        self.update_pair(self.current_source_idx, 
                         self.current_ref_idx, 
                         user_input)
        
    @time_in
    def update_pair(self, source_idx, ref_idx, user_input):
        '''Update the labeller for any source/ref pair.'''
        if self.current_source_idx != source_idx:
            raise NotImplementedError('Update pair not implemented when update' \
                                      ' concerns a source that is not current')
        
        start_time = time.time()
        # Interpret answers
        yes = self.VALID_ANSWERS[user_input] == 'y'
        no = self.VALID_ANSWERS[user_input] == 'n'
        uncertain = self.VALID_ANSWERS[user_input] == 'u'
        forget_row = self.VALID_ANSWERS[user_input] == 'f'  
        assert yes + no + uncertain + forget_row == 1 
        
        #
        pair = (source_idx, ref_idx)
        
        assert pair not in self.labelled_pairs
        self.labelled_pairs.append(pair)
        self.labels[pair] = self.VALID_ANSWERS[user_input]      
        
        if yes:
            labelled_pair = pair
            next_row = True # Skip to next row of source
        
        if no:
            next_row = False
            
        if uncertain:
            raise NotImplementedError('Uncertain is not yet implemented')
            
        if forget_row:
            labelled_pair = (source_idx, '__NO_RESULT') # OG was __FORGET
            next_row = True # OG was True
            
        if not next_row:
            # Try to get next label, otherwise jump        
            try:
                # Change ref row if the ref being labelled is the current ref.
                if ref_idx == self.current_ref_idx:
                    (self.current_ref_idx, self.current_ref_item, self.current_es_score) = next(self.ref_gen)

                # If iterator runs out: next_row = True
                self._update_row_count(False, yes)
                                
            except StopIteration:
                # If no match was found
                labelled_pair = (self.current_source_idx, '__NO_RESULT')
                next_row = True
        
        # NB: not using "else" because there is a chance for next_row to change in previous "if"    
        if next_row:
            # Add rows_labelled_counts
            self._update_row_count(True, yes)            
            
            # Re-score and sort metrics
            self.add_labelled_pair(labelled_pair)
            
            # Update metrics unless the nuew pair is not relevant ('__FORGET')
            # NB: '__NO_RESULT' is relevant for metric re-computing
            if labelled_pair[1] != '__FORGET':
                if self.num_positive_rows_labelled[-1]: # Check that there are positive labels on which to compute metrics           
                    if True: 
                        self._sorta_sort_queries()
                        self._metrics_and_sort()
            
            # Update core queries
            # TODO: look into batching this
            if yes:
                for query in self.single_core_queries:
                    
                    #
                    if source_idx != self.current_source_idx:
                        source_item = self._get_source_item(source_idx)
                    else:
                        source_item = self.current_source_item

                    #
                    if ref_idx != self.current_ref_idx:
                        ref_item = self._get_ref_item(source_idx)
                    else:
                        ref_item = self.current_ref_item
                    
                    query.add_labelled_pair_items(self.es, self.ref_index_name, 
                                    source_item, ref_item)

                        
            # Filter queries
            self.filter_()
            
            # Expand queries
            self.expand()
                
            if self.status == 'ACTIVE':
                # Get new pair
                self._next_row()
                
                self._sanity_check()
                
        
        #        _print_time_in()
        #        print('update_pair: {}'.format(time.time() - start_time))

    @print_name
    def _re_score_history(self, call_next_row, learn=False):        
        """Compute scores for all queries given the current history and 
        parameters.
        
        Perform queries on past labels for `current_queries` and re-scores and 
        sort the queries. Use this after adding filters or after expansion of 
        current_queries. 
        
        If the `learn` paramater is set to True, `current_queries` will be 
        re-initialized and expanded and filtered as they were following manuel
        labelling. If not, `current_queries` will remain (although their score
        and precision might change).
                
        NB: this does not deal with generating the new pair (self._next_row)
        
        Parameters
        ----------
        call_next_row: #TODO: document
        learn: bool
            Whether or not to filter and expand queries while adding labels. 
            If not, only the queries in current_queries will be used.
        """
        
        if (not learn) and (not self.current_queries):
            self.status = 'NO_QUERIES'
            logging.warning('Cannot re-score history: NO_QUERIES')
            return

        # Do not re-score if no labels
        if (not self.num_positive_rows_labelled) or (not self.num_positive_rows_labelled[-1]):
            #self.status = 'NO_ITEMS_TO_LABEL'
            logging.warning('Cannot re-score history: NO_ITEMS_TO_LABEL')
            return
            
        print('WARNING: re-scoring history')
        # Re-initialize queries
        for query in self.current_queries:
            query.reset()
        
        current_source_idx = self.current_source_idx
        current_ref_idx = self.current_ref_idx        
        current_source_item = self.current_source_item
        current_ref_item = self.current_ref_item            
        
        # self._init_queries(self.match_cols, self.columns_to_index)
        
        og_labelled_pairs_match = list(self.labelled_pairs_match)
    
        # TODO: discrepancy between length of self.labels and self.num_rows_labelled
        # TODO: NRL doesn't increase once you have no results once
        
        self.labelled_pairs_match = []
        self.num_rows_labelled = []
        self.num_positive_rows_labelled = []            
        for i, labelled_pair in enumerate(og_labelled_pairs_match):
            
            print('Re-scoring row {0}/{1}'.format(i, len(og_labelled_pairs_match)))
            
            (source_idx, ref_idx) = labelled_pair
            
            # TODO: not re_labelling Nones
            self.current_source_idx = source_idx
            self.current_ref_idx = ref_idx
            
            self.current_source_item = self._fetch_source_item(source_idx)
            # is not needed
            
            self.current_es_score = None
            
            # Fetch data for next row
            results = self.pruned_bulk_search(self.current_queries, 
                                    self.current_source_item, 1) #self.NUM_SEARCH_RESULTS)
            self.add_results(results)
            
            self.add_labelled_pair(labelled_pair)

            last_is_match = ref_idx not in ['__FORGET', '__NO_RESULT']
            self._update_row_count(at_new_row=True, last_is_match=last_is_match) # TODO: No result same behavior in re-score ?

            if learn:
                # Re-score metrics
                # if True: # TODO: use sort_queries
                self._sorta_sort_queries()
                self._metrics_and_sort()
                
                # Filter queries
                self.filter_()
                
                # Expand queries
                self.expand()
                
                # Get new pair
                self._next_row()

        # if True: # TODO: use sort_queries
        self._sorta_sort_queries()
        # Re-score metrics
        self._metrics_and_sort()

        # Go back to original state
        self.current_source_idx = current_source_idx
        self.current_ref_idx = current_ref_idx        
        self.current_source_item = current_source_item
        self.current_ref_item = current_ref_item   
        
        if call_next_row:
            # Fetch data for next row
            results = self.pruned_bulk_search(self.current_queries, 
                            self.current_source_item, self.NUM_SEARCH_RESULTS)
            self.add_results(results)
        
        self._sanity_check()

        
    def answer_is_valid(self, user_input):
        """Check if the user input is valid."""
        return user_input in self.VALID_ANSWERS
    
    def _query_counter_wrapper(func):
        """Decorator to use around filter and expand to print the number of 
        queries before and after transformations.
        """
        def wrapper(self, *args, **kwargs):
            l1 = len(self.current_queries)
            res = func(self, *args, **kwargs)
            l2 = len(self.current_queries)
            
            if l2 > l1:
                word = 'added'
                val = l2 - l1
            else:
                word = 'removed'
                val = l1 - l2
            print('{0}: {1} {2} queries; {3} left'.format(func.__name__, word, val, l2))   
            
            return res
        return wrapper
    
    def _log_wrapper(func):
        """Decorator to use around filter and expand to record changes in 
        `self.log`.
        """
    
        def wrapper(self, *args, **kwargs):
            #self._sort_queries()
            # self._iterative_metrics_and_sort() #TODO: check if this is necessary
            log = dict()
            log['func_name'] = func.__name__
            if self.current_queries:
                best_query = self.current_queries[0]
                log['old_best_query'] = best_query
                log['old_precision'] = best_query.precision
                log['old_recall'] = best_query.recall
                log['old_score'] = best_query.score
            else:
                log['old_best_query'] = None
                log['old_precision'] = None
                log['old_recall'] = None
                log['old_score'] = None          
            
            res = func(self, *args, **kwargs)

            # self._sort_queries()
            self._metrics_and_sort()
            if self.current_queries:    
                best_query = self.current_queries[0]
                log['best_query'] = best_query
                log['precision'] = best_query.precision
                log['recall'] = best_query.recall
                log['score'] = best_query.score
            else:
                log['best_query'] = None
                log['precision'] = None
                log['recall'] = None
                log['score'] = None
                        
            try:
                self.log
            except:
                self.log = []
                
            self.log.append(log)
            
            return res
        return wrapper
    
    @time_in
    @print_name 
    def _prune_analyzers(self, remove_duplicates=False):
        """Prune analzyers on each individual query and remove duplicate queries.
        """
        import time
        a  = time.time()
        for q in self.current_queries:
            q.prune_analyzers()
        b = time.time()
        if remove_duplicates:
            self.current_queries = list(set(self.current_queries))
        c = time.time()
        
        print('step A: {}s / step B: {}s'.format(b-a, c-a))
    
    @time_in
    @print_name
    @_log_wrapper
    @_query_counter_wrapper   
    def filter_by_extended_core(self):
        """Keep the best of each query template for all distinct extended_cores.
        
        Keep the best combination of `boost_level`'s for each `extended_core`.
        """
        queries_by_extended_core = defaultdict(list)
        for query in self.current_queries:
            queries_by_extended_core[query.extended_core].append(query)
            
        self.current_queries = [sorted(queries, key=lambda x: x.score)[-1] \
                                for queries in queries_by_extended_core.values()]
        self._metrics_and_sort()
        
    @time_in
    def filter_(self):
        """Apply filtering on current_queries."""
        FILTER_BY_CORE_IDXS = [10, 20]
        
        if self._nprl() > 1:
            print('FILTERING !')
            self.filter_by_precision()
            self.filter_by_num_keys()
        
        if self._nprl() in FILTER_BY_CORE_IDXS:
            self.filter_by_core()
            
            # TODO: re-score ?
            self._re_score_history(call_next_row=False)
    
        if not self.current_queries:
            self.status = 'NO_QUERIES'
            logging.warning('No more queries after filtering')
        
    
    @time_in
    def expand(self):
        """Use current state to determin whether or not to use query expansion 
        and which method to use.
        """
        EXPAND_BY_CORE_IDXS = {11, 17}
        EXPAND_BY_BOOST_IDXS = {14, 22, 30, 60, 120, 240} # TODO: smarter than that

        assert not bool(EXPAND_BY_CORE_IDXS & EXPAND_BY_BOOST_IDXS)

        try:
            self.already_expanded
        except:
            self.already_expanded = set()

        if self._nprl() in self.already_expanded: # TODO: all this is ugly just to have 1 expansion
            return

        if self._nprl() in EXPAND_BY_CORE_IDXS:
            self.expand_by_core()            
            
        elif self._nprl() in EXPAND_BY_BOOST_IDXS:
            self.expand_by_boost()
            
        else: 
            return
        
        self.already_expanded.add(self._nprl())
        
        assert self.current_source_idx == self.labelled_pairs_match[-1][0]

    
    def _nrl(self):
        """Return the current number of rows labelled."""
        # TODO: num rows_labelled, take care of this
        if self.num_rows_labelled:
            return self.num_rows_labelled[-1]
        else:
            return 0   
    
    def _nprl(self):
        """Return current number of positive matches labelled."""
        # TODO: num rows_labelled, take care of this
        if self.num_rows_labelled:
            return self.num_positive_rows_labelled[-1]
        else:
            return 0

    @time_in
    @print_name
    @_log_wrapper
    @_query_counter_wrapper   
    def filter_by_core(self):
        """Restrict each individual current query to their essential core 
        queries. 
        
        Remove queries for which the core score is too low
        """
        MIN_SCORE = 0.1
        cores = [q.core for q in self.single_core_queries if q.score <= MIN_SCORE]

        self.current_queries = list({query.new_template_restricted(cores, ['must', 'should']) \
                                            for query in self.current_queries})
        self.current_queries = [x for x in self.current_queries if x is not None]
    
    @time_in
    @print_name
    @_log_wrapper
    @_query_counter_wrapper   
    def filter_by_precision(self):
        """Filter current_queries based on their precision."""
        MIN_PRECISION_TAB = [(20, 0.5), (10, 0.4), (5, 0.3)]
        def _min_precision(self):
            """
            Return the minimum precision to keep a query template, according to 
            the number of rows currently labelled
            """
            for min_idx, min_precision in MIN_PRECISION_TAB:
                if self._nprl() >= min_idx:
                    break    
            else:
                min_precision = 0
            return min_precision
        
        precisions = [x.precision for x in self.current_queries]
        sorted_indices = sorted(range(len(precisions)), key=lambda x: precisions[x], reverse=True)

        indices_to_keep = [i for i, x in enumerate(self.current_queries) \
                               if x.precision >= _min_precision(self)]
        indices_to_keep += sorted_indices[len(indices_to_keep): self.MIN_NUM_QUERIES]
        
        # Complex maneuver to avoid copying self.current_queries #TODO: change this ?
        self.current_queries = [x for i, x in enumerate(self.current_queries) \
                                if i in indices_to_keep]
   
    @time_in
    @print_name
    @_log_wrapper
    @_query_counter_wrapper   
    def filter_by_num_keys(self):
        """Keep only the best queries.
        
        Keep only the N best queries. N depends on the number of rows with
        a match found (aka the number of positive labels). The more labels we
        have, the less queries we keep.
        """
        
        MAX_NUM_KEYS_TAB = [(20, 10), (10, 50), (7, 200), (5, 500), (0, 4000)]
        def _max_num_queries(self):
            """
            Max number of labels based on the number of rows currently labelled
            """
            for min_idx, max_num_keys in MAX_NUM_KEYS_TAB[:-1]:
                if self._nprl() >= min_idx:
                    break
            else:
                max_num_keys = MAX_NUM_KEYS_TAB[-1][1]
            return max_num_keys    
        
        # Remove queries according to max number of keys
        self.current_queries = self.current_queries[:_max_num_queries(self)]

    @time_in
    @print_name   
    @_log_wrapper
    @_query_counter_wrapper   
    def expand_by_core(self):
        """Add queries to current_queries by adding fields to current_queries."""
        print('EXPANDING BY CORE')
        MIN_SCORE = 0.7
        cores = [q for q in self.single_core_queries if q.score >= MIN_SCORE]

        self.current_queries = list({x for query in self.current_queries \
                                for x in query.multiply_by_core(cores, ['must'])})
        self._prune_analyzers()
    
        # TODO: Move back into expand
        self._re_score_history(call_next_row=False) # Also sorts results
        self.filter_by_extended_core()
        self.filter_()

    @time_in
    @print_name    
    @_log_wrapper
    @_query_counter_wrapper   
    def expand_by_boost(self):
        """Add queries to current_queries by varying boost levels."""
        print('EXPANDING BY BOOST')        
        self.current_queries = list({x for query in self.current_queries \
                                for x in query.multiply_by_boost(2)})

        # TODO: Move back into expand
        self._re_score_history(call_next_row=False) # Also sorts results
        self.filter_by_extended_core() 
        self.filter_()

    @print_name
    def export_best_params(self):
        """Return the parameters for the best query for matching (use in `es_linker`)."""
  
        params = dict()
        params['index_name'] = self.ref_index_name
        params['queries'] = [{'template': q._as_tuple(), 
                              'thresh': 0,
                              'best_thresh': q.thresh if q.thresh else 0,
                              'expected_precision': q.precision,
                              'expected_recall': q.recall} \
                                  for q in self.current_queries[:self.num_queries_sorted]]
        
        # Sort queries by precision if possible
        assert all(x['expected_precision'] is None for x in params['queries']) \
            or all(x['expected_precision'] is not None for x in params['queries'])
        if params['queries'][0]['expected_precision'] is not None:
            params['queries'] = sorted(params['queries'], 
                          key=lambda x: x['expected_precision'], reverse=True)

        params['must'] = self.must_filters
        params['must_not'] = self.must_not_filters
        
        params['exact_pairs'] = [p for p in self.labelled_pairs if self.labels[p] == 'y']
        params['non_matching_pairs'] = [p for p in self.labelled_pairs if self.labels[p] == 'n']
        params['forgotten_pairs'] = [p for p in self.labelled_pairs if self.labels[p] == 'f']
        
        return params
    
    def write_training(self, file_path): # DONE     
        params = self.export_best_params()
        encoder = MyEncoder()
        with open(file_path, 'w') as w:
            w.write(encoder.encode(params))
    
    @print_name
    def update_musts(self, must_filters, must_not_filters):
        if (not isinstance(must_filters, dict)) or (not isinstance(must_not_filters, dict)):
            raise ValueError('Variables "must" and "must_not" should be dicts' \
                'with keys being column names and values a list of strings')
        self.must_filters = must_filters
        self.must_not_filters = must_not_filters
        
        self.status = 'ACTIVE' # If not 'ACTIVE' this will be fixed in _next_row
        self._re_score_history(call_next_row=True)
        self._sanity_check()
    
    @print_name
    def update_targets(self, t_p, t_r):
        self.TARGET_PRECISION = t_p
        self.TARGET_RECALL = t_r

        # Re-score metrics
        self._metrics_and_sort()
    
#    def next_items(self, max_num_items):
        
    
    def to_emit(self):
        """Creates a dict to be sent to the template."""#TODO: fix this
        dict_to_emit = dict()
        
        # Status
        dict_to_emit['status'] = self.status

        # Info on labeller
        dict_to_emit['t_p'] = self.TARGET_PRECISION
        dict_to_emit['t_r'] = self.TARGET_RECALL
        dict_to_emit['has_previous'] = bool(len(self.labels)) 
        dict_to_emit['must_filters'] = self.must_filters
        dict_to_emit['must_not_filters'] = self.must_not_filters
        
        # Info on labeller (counts)
        dict_to_emit['num_pos'] = sum(self.VALID_ANSWERS[x]=='y' for x in self.labels.values())
        dict_to_emit['num_neg'] = sum(self.VALID_ANSWERS[x]=='n' for x in self.labels.values())
        dict_to_emit['num_unc'] = sum(self.VALID_ANSWERS[x]=='u' for x in self.labels.values())
        dict_to_emit['num_for'] = sum(self.VALID_ANSWERS[x]=='f' for x in self.labels.values())
        
        # Info on current query
        # TODO: on  previous, current_query is no longer valid
        dict_to_emit['query_ranking'] = self.current_query_ranking
        if self.current_query_ranking != -1:
            # best_query = self.current_queries[0]
            # dict_to_emit['query'] = best_query._as_tuple()
            #dict_to_emit['estimated_score'] = best_query.score
            # dict_to_emit['thresh'] = best_query.thresh        

            current_query = self.current_query
            dict_to_emit['c_query'] = current_query._as_tuple()
            dict_to_emit['c_estimated_precision'] = current_query.precision
            dict_to_emit['c_estimated_recall'] = current_query.recall
            dict_to_emit['c_estimated_score'] = current_query.score 
            dict_to_emit['c_thresh'] = current_query.thresh   

        dict_to_emit['estimated_precision'] = self.estimated_precision
        dict_to_emit['estimated_recall'] = self.estimated_recall 
        dict_to_emit['num_queries_sorted'] = self.num_queries_sorted

        # Info on pair
        dict_to_emit['source_idx'] = self.current_source_idx
        dict_to_emit['ref_idx'] = self.current_ref_idx
        
        if isinstance(self.current_source_item, dict):
            csi = self.current_source_item
        else:
            csi = self.current_source_item.to_dict()
        
        dict_to_emit['source_item'] = {'_id': self.current_source_idx, 
                                        '_source': csi}
        
        dict_to_emit['ref_item'] = {'_id': self.current_ref_idx, 
                                    '_score': self.current_es_score,
                                    '_source': self.current_ref_item}
        dict_to_emit['top_ref_items'] = self._ref_rows_for_current_source_row(20, 10**9)
        
        dict_to_emit['es_score'] = self.current_es_score
        dict_to_emit['majority_vote'] = self.majority_vote(10)
        
        # Estimate if the current pair is considered to be a match or not
        # (only if we have access to es_score and current query threshold)
        if self.current_query_ranking != -1:
            if (self.current_es_score is not None) and (current_query.thresh is not None):
                dict_to_emit['estimated_is_match'] = self.current_es_score >= current_query.thresh
            else:
                dict_to_emit['estimated_is_match'] = None
                
        return dict_to_emit
    
    
class SearchLabeller(BasicLabeller):
    """
    Extends the BasicLabeller class by providing tools for custom search.
    """

    def to_dict(self):
        """Returns a dict representation of the instance."""
        custom_searches = self._extract_custom_search()
        if custom_searches:
            x = custom_searches[0]
            self._add_searches([{'_id': x[0], '_source': x[1], '_score': x[2]}], replace_current=True)
        dict_= super().to_dict()
        dict_['custom_search_ref_gen'] = custom_searches[1:]
        return dict_
    
    @classmethod
    def from_dict(cls, es, source, ref_index_name, dict_):
        """Returns an instance of the class using a representation generated 
        by to_dict.
        """
        sl = super(SearchLabeller, cls).from_dict(es, source, ref_index_name, dict_)
        
        res = [{'_id': x[0], '_source': x[1], '_score': x[2]} \
                   for x in dict_['custom_search_ref_gen']]
        
        # Add searches and replace the current top
        if res: 
            assert sl.current_query_ranking == -1
            sl._add_searches(res, replace_current=False)
        return sl    
    
    def custom_search(self, search_params, max_num_results=10):
        """Search for specific items in reference using Elasticsearch.
        
        The items in `search_params` are searched for in the index defined by 
        `self.ref_index_name` and results of the query are added in front of 
        `ref_gen` so as to be proposed as matches before returning to the matches
        proposed using `current_queries`.
        
        Parameters
        ----------
        search_params: dict 
            The values to search for in each columns of the referential.
            Ex: {col1: [val1, val2], col4: [val3]}
        max_num_results: int
            The maximum number of results to append for a search.
        """
        
        EXCLUDED_ANALYZERS = ['n_grams']
        
        # Get search analyzers
        ic = client.IndicesClient(self.es)
        mappings = ic.get_mapping(self.ref_index_name)[self.ref_index_name]['mappings']['structure']['properties']
        analyzers = {col: set(mappings[col].get('fields', {}).keys()) for col in mappings.keys()}
        for col in analyzers:
            analyzers[col].add(mappings[col]['analyzer'])
        
        # Remove exluded analyzers
        analyzers = {col: set(filter(lambda x: x not in EXCLUDED_ANALYZERS, values)) \
                     for col, values in analyzers.items()}
          
        # De-duplicate search_params
        search_params = {col: set(values) for col, values in search_params.items()}
        
        # Create row to search for and query templates
        row = dict()
        tmp = dict()
        for ref_cols, words in search_params.items():
            
            if isinstance(words, str):
                words = [words]
            if isinstance(ref_cols, str):
                ref_cols = [ref_cols]
            
            # Determin the analyzers to use
            col_analyzers = set()
            for k in ref_cols:
                col_analyzers.update(analyzers[k])  
                   
            source_col = sorted(ref_cols)[0]
            tmp[source_col] = []
            for analyzer in col_analyzers:
                tmp_analyzer = []
                for i, word in enumerate(words):
                    # Create separate columns in source for each value to avoid
                    # automatic concatenation
                    source_col_val = source_col + '__' + str(i)
                    row[source_col_val] = word
                
                    s_q_t = ('must', source_col_val, ref_cols, '.'+analyzer, 1)
                    # Tmp stores the query templates that can be used for 
                    
                    tmp_analyzer.append(s_q_t)
                tmp[source_col].append(tmp_analyzer)
        
        template_tuples = [(z for y in x for z in y) for x in itertools.product(*tmp.values())]
        query_templates = [CompoundQueryTemplate(x) for x in template_tuples]
        print('len of query_templates', len(query_templates))   
        
        print('len query_templates[0]', len(query_templates))
        
        if len(query_templates) <= 10:
            search_func = self._bulk_search
        else:
            search_func = self.pruned_bulk_search
        res = search_func(query_templates, row, num_results=max_num_results) # TODO: add global filters ?

        print('\nRow:\n', row)
        print('\nSearch params template:\n', search_params)
        return res
        
    def add_custom_search(self, search_params, max_num_results=10):
        """Search for specific items in reference using Elasticsearch and add 
        to labelling queue.
        
        The items in `search_params` are searched for in the index defined by 
        `self.ref_index_name` and results of the query are added in front of 
        `ref_gen` so as to be proposed as matches before returning to the matches
        proposed using `current_queries`.
        
        This method allows the user to create custom searches on specific words 
        (rather than searching for entire rows of `current_source_item`). This 
        can be useful in two cases: 1) The user is using the labeller to label 
        the entire file by hand; this can help find a match. 2) The labeller
        is having trouble focusing at the beginning of the training; this can 
        help speed up the process for the first few steps.
        
        Parameters
        ----------
        search_params: dict 
            The values to search for in each columns of the referential.
            Ex: {col1: [val1, val2], col4: [val3]}
        max_num_results: int
            The maximum number of results to append for a search.
        """
        
        res = self.custom_search(search_params, max_num_results=10)
        
        # Remove previous results of search
        self.clear_custom_search()
    
        # Flatten
        res = [y for x in res for y in x]
        
        # Order by best of each template #TODO:
        
        print('\nLen res:', len(res))
        
        self._add_searches(res, replace_current=True)
        
    def _add_searches(self, res, replace_current):
        """Update the state of the labeller, prepending to `ref_gen` and setting
        the proper values for current states.
        """
        if res:
            def temp(og_elem, og_ranking, og_query):
                for x in res:
                    # Use this for user_filtered queries
                    self.current_query_ranking = -1
                    self.current_query = None
                    
                    if (self.current_source_idx, x['_id']) not in self.labelled_pairs:
                        yield (x['_id'], x['_source'], x['_score'])
                
                if replace_current:
                    self.current_query_ranking = og_ranking
                    self.current_query = og_query
                    yield og_elem
                
            og_elem = (self.current_ref_idx, self.current_ref_item, self.current_es_score)   
            og_ranking = self.current_query_ranking
            og_query = self.current_query
            
            self.ref_gen = itertools.chain(temp(og_elem, og_ranking, og_query), 
                                           self.ref_gen)
            
            # Change current proposal
            if replace_current:
                (self.current_ref_idx, self.current_ref_item, self.current_es_score) = next(self.ref_gen)
    

    
    def _extract_custom_search(self):
        """Remove all custom searches from `ref_gen` and return them in a list."""

        custom_search_ref_gen = []

        # Add the current item if appropriate
        if self.current_query_ranking == -1:
            custom_search_ref_gen.append((self.current_ref_idx, \
                                self.current_ref_item, self.current_es_score))

        # Add items still in `ref_gen`
        for _ in range(10**6): # Avoiding while True
            try:
                first_elem = next(self.ref_gen)
            except StopIteration:
                def temp():
                    raise StopIteration
                    yield
                self.ref_gen = temp()
                break
                
            if self.current_query_ranking == -1:
                custom_search_ref_gen.append(first_elem)
            else:
                self.ref_gen = itertools.chain([first_elem], self.ref_gen)
                break
        else:
            raise RuntimeError('All elements of ref_gen have ' \
                               'current_query_ranking at -1. This is wrong!')   
            
        return custom_search_ref_gen
    
    def clear_custom_search(self):
        """Remove the elements that were generated by a user search (identified
        by self.current_query_ranking == -1).
        """
        if self.current_query_ranking == -1:
            
            self._extract_custom_search()

            try: 
                (self.current_ref_idx, self.current_ref_item, self.current_es_score) = next(self.ref_gen)
            except StopIteration:
                print(self.current_source_item)
                print('WARNING: no results found for the row above; skipping')
                self._update_row_count(True, False) # TODO: not tested! 
                self._next_row()
                
                

class StatsLabeller(BasicLabeller):
    """Adds logging of performances for each query being tried each time a new
    row is being labelled.
    """
    
    query_definitions = dict()
    query_stats = defaultdict(dict) # {query_id: {step: statsdict, ...}, ...}
    query_order = [] # [[query_id, ...], ...] (of length the number of steps saved)
    current_step = 0
    
    @staticmethod
    def _q_id(q):
        return hashlib.md5((q._as_tuple()).__str__().encode('utf-8')).hexdigest()
    
    @staticmethod
    def _create_query_summary(q, position):
        return {'precision': q.precision,
                'recall': q.recall,
                'score': q.score,
                'position': position}

    @print_name
    def update_stats(self):
        """Update the general stats of all queries with the stats at the 
        current step.
        """
        for pos, q in enumerate(self.current_queries):
            # print('in stats: prec: {}, rec: {}'.format(q.precision, q.recall))
            self.query_definitions.setdefault(self._q_id(q), q._as_tuple())
            self.query_stats[self._q_id(q)][str(self.current_step)] = self._create_query_summary(q, pos)
            
        self.query_order.append([self._q_id(q) for q in self.current_queries])
        
        self.current_step += 1
        
        self._cleanup(5)
        
    def _cleanup(self, num_empty):
        """Remove stats for queries that were not used since `num_empty` steps."""
        all_keys = list(self.query_stats.keys())
        if self.current_step >= num_empty:
            for query_id in all_keys:
                if all(query_id not in list_ for list_ in self.query_order[-num_empty:]):
                    del self.query_stats[query_id]
        
        self.query_order = [[x for x in list_ if x in self.query_stats.keys()] for list_ in self.query_order]
    
    def _next_row(self):
        self.update_stats()
        super()._next_row()
        
    def to_dict(self):
        dict_ = super().to_dict()
        dict_['query_definitions'] = self.query_definitions
        dict_['query_stats'] = self.query_stats
        dict_['query_order'] = self.query_order
        dict_['current_step'] = self.current_step
        return dict_
        
    @classmethod
    def from_dict(cls, es, source, ref_index_name, dict_):
        """Returns an instance of the class using a representation generated 
        by to_dict.
        """
        sl = super(StatsLabeller, cls).from_dict(es, source, ref_index_name, dict_)
        
        sl.query_stats = dict_.get('query_definitions', {})
        sl.query_stats = defaultdict(dict, dict_['query_stats'])
        sl.query_order = dict_['query_order']
        sl.current_step = dict_['current_step']
        return sl
    
    def to_emit(self):
        dict_to_emit = super().to_emit()
        dict_to_emit['query_definitions'] = self.query_definitions
        dict_to_emit['query_stats'] = self.query_stats
        dict_to_emit['query_order'] = self.query_order
        dict_to_emit['current_step'] = self.current_step
        return dict_to_emit
            
class Labeller(SearchLabeller):
    '''Keep this for compatibility.'''
    pass
    
class ConsoleLabeller(Labeller):
    """
    Wrapper around the labeller class for convenient use in the console.    
    """
    
    TABS = ['menu', 'labeller', 'filter']
    
    HELP = '\n*** HELP: What am I supposed to do? ***\n' \
            'The labeller object helps to learn the optimal parameters to use with' \
            ' es_linker. There are two tabs you can switch through: filters and' \
            ' labeller. The labeller tab is used to actually label pairs as match' \
            ' of not_match. The filter tab is used to restrict search withing the' \
            ' referential by adding mandatory or forbidden words in specific columns'
    
    FILTER_INSTRUCTIONS = 'Filter instructions:\n' \
            'Update filters for a given column with the following syntax:\n' \
            '{must_filters or must_not_filters} / {column} / {list_of_elements_to_filter_on}\n' \
            '\n  f.ex: must_not_filters / estab_type / ["kindergarden", "high school"] \n' \
            '  f.ex 2: must_filters / estab_city / ["Paris"]\n' \
            '\nThe first example will force all results from the column estab_type\n' \
            'NOT to include either "kindergarden" or "high school"\n' \
            'The second example will force all results from the column estab_city\n' \
            'NOT to include the word  "Paris"\n'
    
    MENU_INSTRUCTIONS = '#TODO: write menu instructions'
    
    LABELLER_INSTRUCTIONS =     '''Valid answers are:
    (y)es / 1
    (n)o / 0
    (p)revious
    (u)ncertain
    (f)orget
    (q)uit
    (h)elp'''

    
    GENERAL_INSTRUCTIONS = 'Switch tab by entering:\n' \
            '"=labeller", "=menu" or "=filter".\n Quit labeller by typing: "quit"\n' \
            'Help with: "help"'
    
    VALID_TAB_CHANGES = ['=l', '=labeller', '=f', '=filter', '=m', '=menu']

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.current_tab = 'labeller'
        self.finished = False # Are there any more labels

    def process_input(self, user_input):
        """Use the appropriate function to process the user input based on the 
        current tab.
        """        
        if user_input == 'pdb':
            print('You have reached debug mode. Enter "c" to resume')
            import pdb; pdb.set_trace()
            
        elif user_input in ['q', 'quit']:
            logging.warning('Console labeller quit by user')
            self.finished = True
        
        elif user_input in ['h', 'help']:  
            print(self.HELP)
            self.display_instructions()
        
        else:
            if user_input[0] == '=':
                self.change_tab(user_input)
                
            elif self.current_tab == 'labeller':
                if self.status == 'ACTIVE':
                    self.update(user_input)
                else:
                    print('Current status is {0}. Labeller update was not performed'.format(self.status))
                
            elif self.current_tab == 'filter':
                self.update_filter(user_input)
                
            elif self.current_tab == 'menu':
                self.update_menu(user_input)

    def user_input_is_valid(self, user_input):
        if not user_input:
            return False
        elif user_input[0] == '=':
            return user_input[:2] in self.VALID_TAB_CHANGES
        elif user_input in ['q', 'quit', 'pdb', 'h', 'help']:
            return True
        elif self.current_tab == 'labeller':
            return self.answer_is_valid(user_input)
        elif self.current_tab == 'filter':
            return self.filter_user_input_is_valid(user_input)
        elif self.current_tab == 'menu':
            return self.menu_user_input_is_valid(user_input)
            
    def change_tab(self, user_input):
        """Switch to another context tab."""
        if user_input.lower()[:2] == '=l':
            self.current_tab = 'labeller'
            
        elif user_input.lower()[:2] == '=f':
            self.current_tab = 'filter'
            
        elif user_input.lower()[:2] == '=m':
            self.current_tab = 'menu'
    
    def display(self):
        """Show the appropriate display according to the current tab."""
        
        print('\n' + '*'*50)
        print('*** In tab: {0} ***'.format(self.current_tab))
        
        if self.current_tab == 'labeller':
            if self.status == 'ACTIVE':
                self.display_pair()
            else:
                print('>>> Labelling is not possible. Status is: {0}\n'.format(self.status) \
                      + 'You can still update filters (=f).' \
                        ' Type "quit" to exit labeller.')
            
        elif self.current_tab == 'menu':
            self.display_menu()
            
        elif self.current_tab == 'filter':
            self.display_filter()

        if self.finished:
            print('>>> No more pairs to label. You can still update filters.' \
                  'Type "quit" to exit labeller.')

    def display_instructions(self): 
        print('\n*** INSTRUCTIONS for {0} ***'.format(self.current_tab))
        
        if self.current_tab == 'labeller':
            print(self.LABELLER_INSTRUCTIONS)
            
        elif self.current_tab == 'menu':
            print(self.MENU_INSTRUCTIONS)
            
        elif self.current_tab == 'filter':
            print(self.FILTER_INSTRUCTIONS)

        print('\n', self.GENERAL_INSTRUCTIONS)

        if self.finished:
            print('>>> No more pairs to label. You can still update filters.' \
                  'Type "quit" to exit labeller.')


    def display_pair(self):
        """Print current state of labeller and the active pair to label."""
        dict_to_emit = self.to_emit()
    
        if dict_to_emit['query_ranking'] != -1:
            print('Query: {}'.format(dict_to_emit['c_query']))
            print('Query ranking: {}'.format(dict_to_emit['query_ranking']))
            print('Query / Precision: {0}; Recall: {1}; Score: {2}'.format(
                                              dict_to_emit['c_estimated_precision'],
                                              dict_to_emit['c_estimated_recall'],
                                              dict_to_emit['c_estimated_score']))
        
            print('Result ES score: {0}; Query thresh: {1}; Is match: {2}'.format(dict_to_emit['es_score'],
                      dict_to_emit['c_thresh'], dict_to_emit['estimated_is_match']))
            # print('Majority_vote:', dict_to_emit['majority_vote'])
    
        print('\n(S): {0}'.format(dict_to_emit['source_idx']))
        print('(R): {0}'.format(dict_to_emit['ref_idx']))

        for match in self.match_cols:
            print('\n')
            source_cols = match['source']
            if isinstance(source_cols, str):
                source_cols = [source_cols]
            ref_cols = match['ref']
            if isinstance(ref_cols, str):
                ref_cols = [ref_cols]
                
            # TODO: Delete this try except: choose one
            try:
                for source_col in source_cols:
                    print('(S): {0} -> {1}'.format(source_col, dict_to_emit['source_item'][source_col]))
                    
                for ref_col in ref_cols:
                    print('(R): {0} -> {1}'.format(ref_col, dict_to_emit['ref_item'][ref_col]))
            except:
                
                for source_col in source_cols:
                    print('(S): {0} -> {1}'.format(source_col, dict_to_emit['source_item']['_source'][source_col]))
                    
                for ref_col in ref_cols:
                    print('(R): {0} -> {1}'.format(ref_col, dict_to_emit['ref_item']['_source'][ref_col]))


    def display_menu(self):
        print('*** THE MAGICAL CSV MERGE MACHINE ***')
        print('other menu things ...')
    
    def display_filter(self):        
        current_filters = 'Current filters:\n' \
                         + '\n'.join('must_filters / {0} / {1}\n'.format(key, values) \
                                      for key, values in self.must_filters.items()) \
                         + '\n'.join('must_not_filters / {0} / {1}\n'.format(key, values) \
                                      for key, values in self.must_not_filters.items())
    
        print(self.FILTER_INSTRUCTIONS)
        print(current_filters)
    
    def filter_user_input_is_valid(self, user_input):
        values = [x.strip() for x in user_input.split('/', 2)]
        
        return (user_input.count('/') >= 2) \
                and (values[0] in ['must_filters', 'must_not_filters'])     
        
    
    def menu_user_input_is_valid(self, user_input):
        return False 
    
    def update_filter(self, user_input):
        """Change values for labeller permanent filters."""
        
        values = [x.strip() for x in user_input.split('/', 2)]
        
        condition = values[0]
        column = values[1]
        try:
            list_of_strings = eval(values[2])
            if isinstance(list_of_strings, str):
                list_of_strings = [list_of_strings]
        except:
            list_of_strings = [values[2]]
        
        must_filters = self.must_filters
        must_not_filters = self.must_not_filters
        
        if condition == 'must_filters':
            must_filters[column] = list_of_strings
        else:
            must_not_filters[column] = list_of_strings
        
        #
        self.update_musts(must_filters, must_not_filters)
    

    def next_action(self):
        """ """
        display = True
        for x in range(10):
            if display:
                self.display()
            user_input = input('\n > ')
            if self.user_input_is_valid(user_input):
                self.process_input(user_input)
                break
            else:
                print('\n/!\\ INVALID ANSWER /!\\')
                self.display_instructions()
                display = False
        else:
            raise RuntimeError('Too many consecutive wrong orders')
    
    def console_labeller(self, max_num_actions=200):
        for i in range(max_num_actions):
            if self.finished:
                return
            self.next_action()

