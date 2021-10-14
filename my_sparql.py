import re
import ast
import pandas as pd
from setup import PATHS
from fcache.cache import FileCache
from collections import OrderedDict
from json.decoder import JSONDecodeError
from SPARQLWrapper import SPARQLWrapper, JSON
      
def sparqlQuery(query=None):
    if not query==None:
        wrapper = SPARQLWrapper("https://query.wikidata.org/sparql")
        wrapper.setQuery(query)
        wrapper.setReturnFormat(JSON)
        return wrapper.query().convert()
    
def resultAsDict(result=None):
    if result == None:
        return {}
    var = result['head']['vars']
    if (len(var) > 2):
        raise ValueError('More than 2 results vars')
    dic = {}
    for item in result['results']['bindings']:
        dic.update({ extractEntityFromWikidataURL(item[var[0]]['value']) : item[var[1]]['value'] })
    return dic

def resultAsList(result=None):
    if result == None:
        return {}
    var = result['head']['vars']
    if (len(var) > 1):
        raise ValueError('More than 1 results vars')
    l = []
    for item in result['results']['bindings']:
        l.append(extractEntityFromWikidataURL(item[var[0]]['value']))
    return l

def resultAsDataframe(result=None):
    if result == None:
        return None
    df = pd.DataFrame(columns=result['head']['vars'])
    for item in result['results']['bindings']:
        df = df.append(dict(map(lambda kv: (kv[0], kv[1]['value']), item.items())), True)
    return df

def extractEntityFromWikidataURL(url):
    if (not bool(re.match("http://www.wikidata.org/entity/Q[0-9]*", url))):
        return url
    return url[url.rindex("/")+1:]

class PropertyRetreiver:
    """
    """
    
    def __init__(self, identifiers, cache_dir=PATHS['CACHE']):
        identifiers.sort()
        self.__propertyName__ = ''.join(identifiers)
        self.__property__ = '|'.join(['wdt:' + id for id in identifiers])
        self.__cacheFile__ = FileCache('communities.property.' + self.__propertyName__, flag='cs', app_cache_dir=cache_dir)

    def retrieveFor(self, entity=None):
        if entity == None:
            return None
        else:
            if entity not in self.__cacheFile__:
                data = sparqlQuery("SELECT DISTINCT ?" + self.__propertyName__ + " WHERE {" \
                        "   wd:" + entity + " " + self.__property__ + " ?" + self.__propertyName__ +"." \
                        "   SERVICE wikibase:label { bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\". }" \
                        "}" \
                        "LIMIT 100")
                self.__cacheFile__[entity] = resultAsList(data)

            return self.__cacheFile__[entity]

    def close(self):
        self.__cacheFile__.sync()
        self.__cacheFile__.close()

               
            
        