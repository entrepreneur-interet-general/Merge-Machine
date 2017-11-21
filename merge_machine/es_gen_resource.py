#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:58:29 2017

@author: m75380

https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-keep-words-tokenfilter.html
https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export/?disjunctive.country

Valid match if no token on one of both sides

Baariis,Bahliz,Gorad Paryzh,Lungsod ng Paris,
Lutece,Lutetia,Lutetia Parisorum,Lutèce,PAR,Pa-ri,
Paarys,Palika,Paname,Pantruche,Paraeis,Paras,Pari,P
aries,Parigge,Pariggi,Parighji,Parigi,Pariis,Pariisi,Pariizu,Pariižu,Parij,Parijs,Paris,Parisi,Parixe,Pariz,Parize,Parizh,Parizh osh,Parizh',Parizo,Parizs,Pariž,Parys,Paryz,Paryzius,Paryż,Paryžius,Paräis,París,Paríž,Parîs,Parĩ,Parī,Parīze,Paříž,Páras,Párizs,Ville-Lumiere,Ville-Lumière,ba li,barys,pairisa,pali,pari,paris,parys,paryzh,perisa,pryz,pyaris,pyarisa,pyrs,Παρίσι,Горад Парыж,Париж,Париж ош,Парижь,Париз,Парис,Паріж,Փարիզ,פאריז,פריז,باريس,پارىژ,پاريس,پاریس,پیرس,ܦܐܪܝܣ,पॅरिस,पेरिस,पैरिस,প্যারিস,ਪੈਰਿਸ,પૅરિસ,பாரிஸ்,పారిస్,ಪ್ಯಾರಿಸ್,പാരിസ്,ปารีส,ཕ་རི།,ပါရီမြို့,პარიზი,ፓሪስ,ប៉ារីស,パリ,巴黎,파리

"""
from collections import defaultdict
import json
import os

curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

# =============================================================================
# Generate Resource For localiser analyzer
# =============================================================================

elasticsearch_resource_dir = '/etc/elasticsearch'
file_path = os.path.join('resource', 'es_linker', 'geonames-all-cities-with-a-population-1000.json')

# Check that resource is available
if not os.path.isfile(file_path):
    url = 'https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export'
    raise Exception('Missing resource: Download the file from:\n{0}\and place it in:\n{1}'.format(url, file_path))

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

  
file_path_country = os.path.join(elasticsearch_resource_dir, 'es_city_synonyms.txt')      
    
# =============================================================================
# 
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
