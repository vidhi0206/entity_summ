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
#     ESBMVersions.Plus.DBPEDIA_FULL,  # ESBMVersions.Plus.DBPEDIA_FULL or ESBMVersions.V1Dot2.DBPEDIA_TEST_0 or ESBMVersions.V1Dot2.LMDB_TRAIN_1
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
# print(first_root_node)
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
# print(f1_5, f1_10, map_5, map_10)


# # def extract_uri_or_literal(value, last_seg: bool):
# #     match = re.match(r"<(.*?>)", value)
# #     if match and last_seg:
# #         uri = match.group(1)
# #         return uri.split("/")[-1]  # Return last segment of URI
# #     elif match and not last_seg:
# #         uri = match.group(1)
# #         return uri

# #     # Match literal with datatype
# #     if value[0] == '"':
# #         if "^^" in value:
# #             literal_value, datatype = value.rsplit("^^", 1)
# #             datatype = datatype.strip("<>")
# #             if datatype == "http://dbpedia.org/datatype/usDollar":
# #                 return f"${float(literal_value.strip('\"'))}"  # Format as USD
# #             return literal_value.strip('"')  # Return raw literal value

# #         # Match literal with language tag
# #         elif "@" in value:
# #             literal_value, language = value.rsplit("@", 1)
# #             return f"{literal_value.strip('\"')}@{language}"  # Return value with language tag

# #         # Return simple literal
# #         else:
# #             return value.strip('"')
# #     return value


# # print(
# #     ' "Federalistvictory" ',
# #     extract_uri_or_literal("<http://dbpedia.org/resource/Guaram_Mampali> .", False),
# # )


# def sum_last_column(filename):
#     total = 0
#     with open(filename, "r") as file:
#         for line in file:
#             # Split each row by tabs, then split the last item by commas
#             columns = line.strip().split("\t")
#             for column in columns:
#                 # Get the last value in the comma-separated column
#                 last_value = int(column.split(",")[-1])
#                 total += last_value
#     return total


# # Provide the filename
# filename = "totalcounts-1"
# result = sum_last_column(filename)
# print(f"Sum of the last column: {result}")
def parse_triple(triple: str) -> tuple[str, str, str]:
    # triple = triple.rstrip(" .")  # Remove trailing space and dot
    subject, predicate, obj = triple.split(" ", 2)  # Split into three parts
    return subject.strip("<>"), predicate.strip("<>"), obj.strip("<>")


def extract_obj(value):
    match = re.match(r"<(.*?)>", value)
    if match:
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


def convert_triples_to_dict(
    triples: list[str],
) -> dict[str, list[tuple[str, str, str]]]:  # convert the triple into dictionary format
    result = {}
    for triple in triples:
        print(triple)
        subject, predicate, obj = parse_triple(triple)
        if subject not in result:
            result[subject] = []
        result[subject].append((subject, predicate, obj))
    return result


tpl = "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://dbpedia.org/ontology/team> <http://dbpedia.org/resource/Chonburi_F.C.> ."
val = "<http://dbpedia.org/resource/Chonburi_F.C.>"
# print(parse_triple(tpl))
# print(extract_obj(val))
tpl_list = [
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://dbpedia.org/ontology/team> http://dbpedia.org/resource/Chonburi_F.C.",
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://dbpedia.org/ontology/team> http://dbpedia.org/resource/Muangthong_United_F.C.",
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://dbpedia.org/ontology/location> http://dbpedia.org/resource/National_Stadium_(Thailand)",
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://dbpedia.org/ontology/city> http://dbpedia.org/resource/Bangkok",
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://dbpedia.org/ontology/FootballMatch",
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Event",
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.wikidata.org/entity/Q1656682",
    "<http://dbpedia.org/resource/2011_Kor_Royal_Cup> <http://xmlns.com/foaf/0.1/name> 2011 Kor Royal Cup@en",
]
print(convert_triples_to_dict(tpl_list))
