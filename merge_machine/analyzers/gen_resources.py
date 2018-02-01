#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:58:29 2017

@author: m75380

Run this script in sudo to generate resource files for Elasticsearch analyzers 

https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-keep-words-tokenfilter.html
https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export/?disjunctive.country
"""
from collections import defaultdict
import json
import logging
import os
import requests
import tempfile
import urllib
import urllib.request


from elasticsearch import Elasticsearch
from unidecode import unidecode

logging.basicConfig(level=logging.INFO)

curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

# =============================================================================
# Messages
# =============================================================================

skip_msg = 'Skipping resource generation for analyzer "{}" (files already '\
           'exist). You can override this by setting the "force" argument to True'

#==============================================================================
# Helper functions
#==============================================================================

def demand_confirmation(size, url, file_path):
    """Ask for user input before downloading a file.
    
    Parameters
    ----------
    size: str
        The size of the file about to be downloaded.
    url: str
        URL from which we download resources
    file_path: str
        The path to which the file will be written.
    
    Returns
    -------
    bool:
        Whether or not the user confirmed download
    """
    user_confirm = input('This script will download temporary resources files' \
                 '({}) from {} and place them in {}.'.format(size, url, file_path) \
                 + '\nYou can delete them afterwards. \n' \
                 'Shall we proceed? y(es)/n(o) (defaults to yes)\n>')
    if user_confirm.lower() in ['yes', 'y', '']:
        return True
    return False

def _get_default_conf_dir():
    """Return the default directory for elasticsearch configuration (in which to)
    write resource files (usually /etc/elasticsearch).
    
    Returns
    -------
    str or None:
        Return None when there are multiple directories (multiple nodes). 
        Otherwise, return the default conf directory path.
    """
    nodes = Elasticsearch().nodes.info()['nodes']
    resource_dirs = list({node['settings']['default']['path']['conf'] \
                         for node in nodes.values()})
    if len(resource_dirs) > 1:
        elasticsearch_resource_dir = None
    else:
        elasticsearch_resource_dir = resource_dirs[0]
    return elasticsearch_resource_dir

        
def write_keep_syn(name_alt_gen, file_path_keep, file_path_syn,
                   asciifolding=True, chars_to_replace=['-', "'"]):
    '''Write both a filter (keep) and synonym file to use as Elasticsearch
    resources.
    
    Both files are used in the context of categorical filtering. For example, 
    if you want to constrain a search to be geographically consistant with 
    respect to country name, there are two steps in the analyzer: 
        1) Filter out the country names
        2) Translate all country names to a chosen language to make comparison
            possible accross languages
    This functions takes care of generating both files from a synonym generator.
    
    Parameters
    ----------
    name_alt_gen: generator of tuples of shape `(name, list_of_alternates)`
        A genarator that associates to a names it's alternate versions 
        (synonyms for example).
    file_path_keep: str
        Path to the resource file used for filtering.
    file_path_syn: str
        Path to the resource file used for synonyms (translating).
    '''
    
    logging.info('Writing to {} and {}'.format(file_path_syn, file_path_keep))
    with open(file_path_syn, 'w') as w_syn, \
         open(file_path_keep, 'w') as w_keep:
        for name, alternates in name_alt_gen:
            alternates = set(alternates)
            alternates.add(name)
            if asciifolding:
                alternates = [unidecode(x) for x in alternates]
            for char in chars_to_replace:
                alternates = [x.replace(char, ' ') for x in alternates]
                
            alternates = {x.strip(', ') for x in alternates}
            alternates = {x.replace(' ', '_') for x in alternates}
            alternates = {x for x in alternates if x}
            name = name.replace(' ', '_')
            
            # sea biscuit, sea biscit => seabiscuit
            if alternates:
                string = ', '.join(alternates) + ' => ' + name + '\n'
                w_syn.write(string)
            
            w_keep.write(name + '\n')
            for alternate in alternates:
                w_keep.write(alternate + '\n')

#==============================================================================
# Resource generation functions
#==============================================================================

def gen_resource_city(elasticsearch_resource_dir, force=False):
    '''Generate resource files for the city analyzer'''
    
    # Paths to synonym and keep files for the city analyzer 
    file_path_keep = os.path.join(elasticsearch_resource_dir, 'city_keep.txt')
    file_path_syn = os.path.join(elasticsearch_resource_dir, 'city_synonyms.txt')

    if os.path.isfile(file_path_keep) and os.path.isfile(file_path_syn) and not force:
        logging.warning(skip_msg.format('city'))
        return

    url = 'https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000@public/download/?format=json&timezone=Europe/Berlin&use_labels_for_header=true'
    handle, file_path = tempfile.mkstemp()
    if not demand_confirmation('100M', url, file_path):
        logging.warning('Aborting resource generation for "city" analyzer')
        return

    # TODO: change for CSV, take less space...
    # View data here https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export
    logging.info('Downloading resource (100M) from:\n{0}\nWriting to:\n{1}\nThis may take some time...'.format(url, file_path))

    urllib.request.urlretrieve(url, file_path)
    with open(file_path) as f:
        res = json.load(f)
    
    # no_country_count = 0
    no_alternate_count = 0
    
    # countries = set()
    # cities_to_countries = defaultdict(set)
    name_to_alternates = defaultdict(set)
    for i, row in enumerate(res):
        if i%50000 == 0:
            logging.info('Generating city resource {0}/{1}'.format(i, len(res)))
            
        my_row = row['fields']
        name = my_row['name']
        
        if 'alternate_names' in my_row:
            alternates = my_row['alternate_names'].split(',')
        else:
            alternates = []
            no_alternate_count += 1
            
        if alternates:
            name_to_alternates[name].update(alternates)
        
    name_alt_gen = name_to_alternates.items()
    
    # Create the resource files (needs sudo rights)
    write_keep_syn(name_alt_gen, file_path_keep, file_path_syn)
    
# =============================================================================
# 
# =============================================================================

def gen_resource_organization(elasticsearch_resource_dir, force=False):
    """Generate resource files for the organization analyzer."""

    file_path_keep = os.path.join(elasticsearch_resource_dir, 'es_organization_keep.txt') 
    file_path_syn = os.path.join(elasticsearch_resource_dir, 'es_organization_synonyms.txt')
    
    if os.path.isfile(file_path_keep) and os.path.isfile(file_path_syn) and not force:
        logging.warning(skip_msg.format('organization'))
        return
    
    org_data = [['lycée', 'lyc', 'préparatoire', 'prépa', 'cpge'], 
     ['collège'], 
     ['université', 'iut'],
     ['école'], 
     ['maternelle'],
     ['primaire'], 
     ['laboratoire', 'labo'], 
     ['institut', 'département']]
    
        
    write_keep_syn(zip([x[0] for x in org_data], org_data), file_path_keep, file_path_syn)
        
    
    [['agence', 'département'], 
     ['association', 'groupement'], 
     ['fédération']]
    
def gen_resource_country(elasticsearch_resource_dir, force=False):
    """Generate the resource files  for the country analyzer."""

    file_path_keep = os.path.join(elasticsearch_resource_dir, 'countries_keep.txt') 
    file_path_syn = os.path.join(elasticsearch_resource_dir, 'countries_synonyms.txt')

    if os.path.isfile(file_path_keep) and os.path.isfile(file_path_syn) and not force:
        logging.warning(skip_msg.format('city'))
        return
    
    url = 'https://raw.githubusercontent.com/mledoze/countries/master/dist/countries.json'
    
    countries = json.loads(requests.get(url).content.decode('utf-8'))
    
    name_alt = []
    for country in countries:
        name = country['cca3']
        
        alt = [country['name']['common']]
        
        # Add native names to alternates
        if country['name']['native']:
            alt.extend([val['common'] for key, val in country['name']['native'].items()] \
                       + [val['official'] for key, val in country['name']['native'].items()])
        
        # Add translations to alternates
        if country['translations']:
            alt.extend([val['common'] for key, val in country['translations'].items()] \
                       + [val['official'] for key, val in country['translations'].items()])
            
        # Add altSpellings, and country codes to alternates
        alt.extend(country['altSpellings'])
        alt.extend(country[key] for key in ['cca2', 'cca3', 'ccn3'])
    
        alt = list(set(alt))
        
        name_alt.append((name, alt))
    
    write_keep_syn(name_alt, file_path_keep, file_path_syn)


def generate_resources(analyzers, elasticsearch_resource_dir=None, force=False):
    """Generate resources for a all analyzers specified within a list.
    
    Parameters
    ----------
    analyzers: list of str
        List of analyzers for which to generate resources.
    elasticsearch_resource_dir: str
       Path to the directory in which to write the Elasticsearch config files.
    force: bool
    """
    
    if elasticsearch_resource_dir is None:
        elasticsearch_resource_dir = _get_default_conf_dir()
        logging.info('Writing Elasticsearch config files to {}'.format(elasticsearch_resource_dir))

    for analyzer in analyzers:
        if analyzer in RESOURCE_GENERATORS:
            RESOURCE_GENERATORS[analyzer](elasticsearch_resource_dir, force)
        else:
            raise Warning('No resource generator for analyzer: {}'.format(analyzer))

RESOURCE_GENERATORS = {'city': gen_resource_city, 
                       'country': gen_resource_country,
                       'organization': gen_resource_organization}

if __name__ == '__main__':
    
    import argparse
    DESCRIPTION = 'Fetch, transform and write resources for Elasticearch' \
                  ' analyzers (synonyms, filters, etc.).'
    EPILOG = '/!\ Writing to Elasticsearch resource directories might require' \
             ' running this script with sudo permissions, or changing the' \
             ' rights for your resource directory.'
        
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('analyzers', nargs='*', 
                        help='List of analyzers for which to create resources.' \
                ' Choose from:\n{}'.format(list(RESOURCE_GENERATORS.keys())), 
                        default=list(RESOURCE_GENERATORS.keys()))
    parser.add_argument('-d', '--es-resource-dir',  
                        help='Elasticsearch resource directory',
                        default=_get_default_conf_dir())
    parser.add_argument('-f', '--force',
                        action='store_true',
                        help='Force re-creation of existing files.')
            
    args = parser.parse_args()

    elasticsearch_resource_dir = args.es_resource_dir
    if elasticsearch_resource_dir is None:
        raise RuntimeError('Multiple resource directories were found for'\
                           ' Elasticsearch in multiple nodes. Use the -d flag' \
                           ' to specify where to write the resource files')
    
    generate_resources(args.analyzers, elasticsearch_resource_dir, args.force)
