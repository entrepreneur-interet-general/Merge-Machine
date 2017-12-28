#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:58:29 2017

@author: m75380

Run this script in sudo to generate resource files for Elasticsearch analyzers 


https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-keep-words-tokenfilter.html
https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export/?disjunctive.country

Valid match if no token on one of both sides

"""
from collections import defaultdict
import json
import os
import requests
import urllib
import urllib.request

from unidecode import unidecode

curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

elasticsearch_resource_dir = '/etc/elasticsearch'

#==============================================================================
# Helper functions
#==============================================================================

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

def gen_resource_city():
    '''Generate resource files for the city analyzer'''
    
    file_path = os.path.join('resource', 'es_linker', 
                             'geonames-all-cities-with-a-population-1000.json')
    # Check that resource is available
    if not os.path.isfile(file_path):
        if not os.path.isdir(os.path.split(file_path)[0]):
            os.makedirs(os.path.split(file_path)[0])
        # TODO: change for CSV, take less space...
        # View data here https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export
        url = 'https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000@public/download/?format=json&timezone=Europe/Berlin&use_labels_for_header=true'
        print('Downloading resource (100M) from:\n{0}\nWriting to:\n{1}\nThis may take some time...'.format(url, file_path))
        urllib.request.urlretrieve(url, file_path)
        
    with open(file_path) as f:
        res = json.load(f)
    
    # no_country_count = 0
    no_alternate_count = 0
    
    # countries = set()
    # cities_to_countries = defaultdict(set)
    name_to_alternates = defaultdict(set)
    for i, row in enumerate(res):
        if i%10000 == 0:
            print('Did {0}/{1}'.format(i, len(res)))
            
        my_row = row['fields']
        name = my_row['name']
        
        if 'alternate_names' in my_row:
            alternates = my_row['alternate_names'].split(',')
        else:
            alternates = []
            no_alternate_count += 1
            
        if alternates:
            name_to_alternates[name].update(alternates)
        
        
        #        if 'country' in my_row:
        #            country = my_row['country']
        #    
        #            countries.add(country)
        #            for city in [name] + alternates:
        #                cities_to_countries[city].add(country)
        #        else:
        #            no_country_count += 1
    
    name_alt_gen = name_to_alternates.items()
    
    
    # Paths to synonym and keep files for the city analyzer 
    file_path_keep = os.path.join(elasticsearch_resource_dir, 'city_keep.txt')
    file_path_syn = os.path.join(elasticsearch_resource_dir, 'city_synonyms.txt')
    
    # Create the resource files (needs sudo rights)
    write_keep_syn(name_alt_gen, file_path_keep, file_path_syn)
    
# =============================================================================
# 
# =============================================================================

def gen_resource_organization():
    """Generate resource files for the organization analyzer."""

    file_path_keep = os.path.join(elasticsearch_resource_dir, 'es_organization_keep.txt') 
    file_path_syn = os.path.join(elasticsearch_resource_dir, 'es_organization_synonyms.txt')
    
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
    
    
    #import requests
    #url = 'https://raw.githubusercontent.com/David-Haim/CountriesToCitiesJSON/master/countriesToCities.json'
    #
    #try:
    #    country_to_cities
    #except:
    #    country_to_cities = json.loads(requests.get(url).content.decode('utf-8'))
    #
    #all_cities = set()
    #city_to_countries = defaultdict(list)
    #for country, cities in country_to_cities.items():
    #    
    #    if country != '':
    #        all_cities.update(cities)
    #        for city in set(cities):
    #            city_to_countries[city].append(country)
    #    
    #    


def gen_resource_country():
    """Generate the resource files  for the country analyzer."""

    file_path_keep = os.path.join(elasticsearch_resource_dir, 'countries_keep.txt') 
    file_path_syn = os.path.join(elasticsearch_resource_dir, 'countries_synonyms.txt')
    
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
    
if __name__ == '__main__':
#    gen_resource_city()
#    gen_resource_organization()
    gen_resource_city()
    gen_resource_country()
