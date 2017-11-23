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
import urllib
import urllib.request

curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

# =============================================================================
# Generate Resource For localiser analyzer
# =============================================================================

elasticsearch_resource_dir = '/etc/elasticsearch'

file_path = os.path.join('resource', 'es_linker', 'geonames-all-cities-with-a-population-1000.json')

# Check that resource is available
if not os.path.isfile(file_path):
    if not os.path.isdir(os.path.split(file_path)[0]):
        os.makedirs(os.path.split(file_path)[0])
    # View data here https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export
    url = 'https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000@public/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true'
    print('Downloading resource (100M) from:\n{0}\nWriting to:\n{1}\nThis may take some time...'.format(url, file_path))
    urllib.request.urlretrieve(url, file_path)
    
with open(file_path) as f:
    res = json.load(f)

no_country_count = 0
no_alternate_count = 0

countries = set()
cities_to_countries = defaultdict(set)
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
        
    if 'country' in my_row:
        
        if alternates:
            name_to_alternates[name].update(alternates)
        
        country = my_row['country']

        countries.add(country)
        for city in [name] + alternates:
            cities_to_countries[city].add(country)
    else:
        no_country_count += 1

def write_keep_syn(name_alt_gen, file_path_keep, file_path_syn):
    with open(file_path_syn, 'w') as w_syn, \
         open(file_path_keep, 'w') as w_keep:
        for name, alternates in name_alt_gen:
            # sea biscuit, sea biscit => seabiscuit
            if name in alternates:
                alternates = set(alternates)
                alternates.remove(name)
            if alternates:
                string = ', '.join(alternates) + ' => ' + name + '\n'
                w_syn.write(string)
            w_keep.write(name + '\n')
            for alternate in alternates:
                w_keep.write(alternate + '\n')

# Generate synonym and cities to keep and write to ES dir (needs sudo rights)
file_path_keep = os.path.join(elasticsearch_resource_dir, 'es_city_keep.txt')
file_path_syn = os.path.join(elasticsearch_resource_dir, 'es_city_synonyms.txt')

name_alt_gen = name_to_alternates.items()
write_keep_syn(name_alt_gen, file_path_keep, file_path_syn)

file_path_country = os.path.join(elasticsearch_resource_dir, 'es_country_synonyms.txt')      


# =============================================================================
# 
# =============================================================================



# =============================================================================
# Generate Resource For organization analyzer
# =============================================================================
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
