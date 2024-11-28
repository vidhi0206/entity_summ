import os
import requests
import gzip
import pandas as pd
import numpy as np
import math
import re
import pulp
from similarity import sim
from heirarchy import build_hierarchy_map, is_related_property, is_related_value
import mpmath
from wikes_toolkit import WikESToolkit, ESBMGraph, ESBMVersions

# from wikes_toolkit import WikESToolkit, ESBMGraph, ESBMVersions

# toolkit = WikESToolkit()
# G = toolkit.load_graph(
#     ESBMGraph,
#     ESBMVersions.V1Dot2.DBPEDIA_FULL,  # ESBMVersions.Plus.DBPEDIA_FULL or ESBMVersions.V1Dot2.DBPEDIA_TEST_0 or ESBMVersions.V1Dot2.LMDB_TRAIN_1
#     entity_formatter=lambda e: e.identifier,
# )
# print(G)
# root_nodes = G.root_entities()
# first_root_node = G.root_entity_ids()[0]
# nodes = G.entities()
# edges = G.triples()
# labels = G.predicates()
# number_of_nodes = G.total_entities()
# number_of_directed_edges = G.total_triples()
# node = G.fetch_entity("http://dbpedia.org/resource/Adrian_Griffin")
# node_degree = G.degree("http://dbpedia.org/resource/Adrian_Griffin")
# gold_top5_0 = G.gold_top_5(node, 0)
# print(gold_top5_0)
# gold_top10_0 = G.gold_top_10(node, 0)
# neighbors = G.neighbors(node)
# # print(neighbors)
# G.mark_triples_as_summaries(
#     "http://dbpedia.org/resource/3WAY_FM",
#     [
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://purl.org/dc/terms/subject",
#             "http://dbpedia.org/resource/Category:Radio_stations_in_Victoria",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://xmlns.com/foaf/0.1/homepage",
#             "http://3wayfm.org.au",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://dbpedia.org/ontology/broadcastArea",
#             "http://dbpedia.org/resource/Victoria_(Australia)",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://dbpedia.org/ontology/callsignMeaning",
#             "3 - Victoria",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://xmlns.com/foaf/0.1/name",
#             "3WAY FM@en",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://www.w3.org/2000/01/rdf-schema#label",
#             "3WAY FM@en",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://dbpedia.org/ontology/callsignMeaning",
#             "Warrnambool And You",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://dbpedia.org/ontology/broadcastArea",
#             "http://dbpedia.org/resource/Warrnambool",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
#             "http://schema.org/RadioStation",
#         ),
#         (
#             "http://dbpedia.org/resource/3WAY_FM",
#             "http://purl.org/dc/terms/subject",
#             "http://dbpedia.org/resource/Category:Radio_stations_established_in_1990",
#         ),
#     ],
# )

# # G.clear_summaries()

# # G.mark_triples_as_summaries(
# #     "http://dbpedia.org/resource/3WAY_FM",
# #     [
# #         (
# #             "http://dbpedia.org/resource/3WAY_FM",
# #             "http://purl.org/dc/terms/subject",
# #             "http://dbpedia.org/resource/Category:Radio_stations_in_Victoria",
# #         )
# #     ],
# # )

# G.mark_triples_as_summaries(
#     "http://dbpedia.org/resource/Adrian_Griffin",
#     [
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://dbpedia.org/ontology/activeYearsEndYear",
#             "2008",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://purl.org/dc/terms/subject",
#             "http://dbpedia.org/resource/Category:Orlando_Magic_assistant_coaches",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://dbpedia.org/ontology/birthDate",
#             "1974-07-04",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://dbpedia.org/ontology/height",
#             "1.9558",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://dbpedia.org/ontology/weight",
#             "98431.2",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://dbpedia.org/ontology/team",
#             "http://dbpedia.org/resource/Orlando_Magic",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://purl.org/dc/terms/subject",
#             "http://dbpedia.org/resource/Category:Basketball_players_from_Kansas",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
#             "http://dbpedia.org/class/yago/BasketballPlayersFromKansas",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
#             "http://dbpedia.org/class/yago/BasketballCoach109841955",
#         ),
#         (
#             "http://dbpedia.org/resource/Adrian_Griffin",
#             "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
#             "http://dbpedia.org/class/yago/BasketballPlayer109842047",
#         ),
#     ],
# )
# print(G._predicted_summaries)
# f1_5 = G.f1_score(5)
# f1_10 = G.f1_score(10)
# map_5 = G.map_score(5)
# map_10 = G.map_score(10)
# print(f1_5, f1_10)


def extract_uri_or_literal(value, last_seg: bool):
    match = re.match(r"<(.*?>)", value)
    if match and last_seg:
        uri = match.group(1)
        return uri.split("/")[-1]  # Return last segment of URI
    elif match and not last_seg:
        uri = match.group(1)
        return uri

    # Match literal with datatype
    if value[0] == '"':
        if "^^" in value:
            literal_value, datatype = value.rsplit("^^", 1)
            datatype = datatype.strip("<>")
            if datatype == "http://dbpedia.org/datatype/usDollar":
                return f"${float(literal_value.strip('\"'))}"  # Format as USD
            return literal_value.strip('"')  # Return raw literal value

        # Match literal with language tag
        elif "@" in value:
            literal_value, language = value.rsplit("@", 1)
            return f"{literal_value.strip('\"')}@{language}"  # Return value with language tag

        # Return simple literal
        else:
            return value.strip('"')
    return value


print(
    ' "Federalistvictory" ',
    extract_uri_or_literal("<http://dbpedia.org/resource/Guaram_Mampali> .", False),
)
