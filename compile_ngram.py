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

B = 4541627  # total number of books
total_number_of_pages = 2441898561
eps = 10**-6
k = 3


def download_file(url, filename):
    # Download the file
    print(f"Downloading file from {url}...")
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"File '{filename}' downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return


def delete_file(filename):
    try:
        # Delete the file
        if os.path.exists(filename):
            os.remove(filename)
            print(f"File '{filename}' deleted successfully.")
        else:
            print(f"File '{filename}' does not exist, so it could not be deleted.")

    except Exception as e:
        print(f"An error occurred: {e}")


def read_gz_csv(file_path):
    with gzip.open(file_path, "rt") as gz_file:
        df = pd.read_csv(gz_file, sep=r"[ \t]+", engine="python", header=None)
    return df


def familiarityFunction(k, B):
    if B == 0 or k == 0:
        # print("error")
        return 0
    return math.log(k + 1) / math.log(B + 1)


def expectation(b_f, k, B):
    # # print(b_f, M, B)
    # Max = min(b_f, M)
    # Min, expectation, factor = 1, 0, 0
    # common_factor = (
    #     math.lgamma(b_f + 1)
    #     + math.lgamma(B - b_f)
    #     + math.lgamma(M)
    #     + math.lgamma(B - M + 1)
    #     - math.lgamma(M)
    # )
    # for k in range(Min, Max ):
    #     factor = (
    #         math.lgamma(k)
    #         + math.lgamma(M - k)
    #         + math.lgamma(b_f - k + 1)
    #         + math.lgamma(B - M - b_f + k)
    #     )
    #     # print(factor, common_factor)
    #     factor = common_factor - factor
    #     # print(factor)
    #     expectation += math.exp(factor) * familiarityFunction(k)
    # expectation = 12345
    # Convert inputs to mpmath-compatible types for arbitrary precision
    # b_f = float(b_f)  # Convert to native float
    # M = float(M)
    # B = float(B)

    # # Convert to mpmath's arbitrary precision format
    # b_f = mpmath.mpf(b_f)
    # M = mpmath.mpf(M)
    # B = mpmath.mpf(B)

    # Max = min(b_f, M)
    # Min, expectation = mpmath.mpf(0), mpmath.mpf(0)
    # # print(B - b_f + 1)
    # common_factor = (
    #     mpmath.loggamma(b_f + 2) + mpmath.loggamma(B - b_f + 1)
    #     if B - b_f > 0
    #     else (
    #         mpmath.mpf(0)
    #         + mpmath.loggamma(M + 1)  # Inclusive range adjustment
    #         + mpmath.loggamma(B - M + 2)
    #         if B - M > 0
    #         else mpmath.mpf(0) - mpmath.loggamma(B + 2)
    #     )
    # )

    # for k in range(int(Min), int(Max + 1)):
    #     factor = (
    #         mpmath.loggamma(k + 1) + mpmath.loggamma(M - k + 2)
    #         if M - k > 0
    #         else (
    #             mpmath.mpf(0) + mpmath.loggamma(b_f - k + 1)
    #             if b_f - k > 0
    #             else (
    #                 mpmath.mpf(0) + mpmath.loggamma(B - M - b_f + k + 1)
    #                 if B - M - b_f + k > 0
    #                 else mpmath.mpf(0)
    #             )
    #         )
    #     )
    #     # print(common_factor, factor)
    #     factor = common_factor - factor
    #     expectation += mpmath.exp(factor) * familiarityFunction(k, B)
    #     # print(expectation)
    # return float(expectation)
    expectation = (b_f / B) * familiarityFunction(k, B)
    return expectation


def logP(df, prop, num_unique_prop):
    filtered_df = df[df[1] == prop]
    filtered_df = filtered_df[0].unique()
    prob_feature = (-1) * math.log(len(filtered_df) / (num_unique_prop + eps))
    return prob_feature


def calc_n_gram_vols(filtered_1_gram_path, filtered_2_gram_path, prop):
    one_grams = []
    two_grams = []
    min_page_count = float("inf")
    min_vol_count = float("inf")
    split_words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", prop)
    one_grams.extend(split_words)
    two_grams.extend(
        [" ".join(split_words[i : i + 2]) for i in range(len(split_words) - 1)]
    )
    # print(one_grams, two_grams)
    filtered_1_gram = pd.read_csv(
        filtered_1_gram_path, sep=r"[ \t]+", header=None, engine="python"
    )
    for one_gram in one_grams:
        filtered_df = filtered_1_gram[filtered_1_gram[0].astype(str) == one_gram]
        page_count = (
            filtered_df.iloc[:, 2].sum() if len(filtered_1_gram.columns) > 2 else 0
        )
        vol_count = (
            filtered_df.iloc[:, 3].sum() if len(filtered_1_gram.columns) > 3 else 0
        )
        if page_count < min_page_count:
            min_page_count = page_count
        if vol_count < min_vol_count:
            min_vol_count = vol_count

    for two_gram in two_grams:
        first_word, second_word = two_gram.split()
        filtered_2_gram = pd.read_csv(
            filtered_2_gram_path, sep=r"[ \t]+", header=None, engine="python"
        )
        filtered_df = filtered_2_gram[
            (filtered_2_gram[0].astype(str) == first_word)
            & (filtered_2_gram[1].astype(str) == second_word)
        ]
        page_count = (
            filtered_df.iloc[:, 3].sum() if len(filtered_2_gram.columns) > 3 else 0
        )
        vol_count = (
            filtered_df.iloc[:, 4].sum() if len(filtered_2_gram.columns) > 4 else 0
        )

        if page_count < min_page_count:
            min_page_count = page_count
        if vol_count < min_vol_count:
            min_vol_count = vol_count

    return min_page_count, min_vol_count


def filter_df_1gram(unique_prop, url_list_2_gram, url_list_1_gram):
    filename1 = "googlebooks-eng-all-1-1gram-20120701"
    one_grams = []
    for prop in unique_prop:
        split_words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", prop)
        one_grams.extend(split_words)

    # print(one_grams,two_grams)
    filtered_df_1_gram = pd.DataFrame()
    for url in url_list_1_gram:
        download_file(url, filename1)
        with gzip.open(filename1, "rt") as gz_file:
            for df in pd.read_csv(
                gz_file, sep=r"[ \t]+", engine="python", header=None, chunksize=100000
            ):
                for one_gram in one_grams:
                    current_filtered_df = df[
                        df[0].astype(str).str.lower() == one_gram.lower()
                    ]
                    filtered_df_1_gram = pd.concat(
                        [filtered_df_1_gram, current_filtered_df]
                    )
        delete_file(filename1)
    filtered_df_1_gram.to_csv(
        "filtered_1_gram.csv", sep="\t", encoding="utf-8", index=False
    )


def filter_df_2gram(unique_prop, url_list_2_gram):
    filename1 = "googlebooks-eng-all-1-1gram-20120701"
    one_grams = []
    two_grams = []
    for prop in unique_prop:
        split_words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", prop)
        one_grams.extend(split_words)
        two_grams.extend(
            [" ".join(split_words[i : i + 2]) for i in range(len(split_words) - 1)]
        )

    two_grams_lower = [word.lower() for word in two_grams]
    # print(two_grams_lower)

    two_words_pattern = re.compile(r"^\s*(\w+)\s+(\w+)")

    with open("filtered_output_2gram.csv", "a") as output_file:
        for url in url_list_2_gram:
            download_file(url, filename1)
            with gzip.open(filename1, "rt") as gz_file:
                for line in gz_file:
                    # Extract the first two words from the line using regex
                    match = two_words_pattern.match(line)
                    # print(match," before if")
                    if match:
                        # print(match," after if")
                        first_word, second_word = match.groups()
                        # Join the two words into a bigram
                        bigram = f"{first_word} {second_word}"
                        # Check if the bigram is in the list of 2-grams
                        if bigram.lower() in two_grams_lower:
                            # print(line)
                            output_file.write(line)
            delete_file(filename1)


def support(propm, propn, df):
    has_propn = df[df[1] == propn][0]
    has_propm = df[df[1] == propm][0]
    intersection = set(has_propm).intersection(set(has_propn))
    # print(intersection)
    supp = len(intersection)
    return supp


def support_plus(propm, propn, df, theta):
    has_propn = df[df[1] == propn][0]
    has_propm = df[df[1] == propm][0]
    intersection = set(has_propm).intersection(set(has_propn))
    supp_plus = 0

    for e in intersection:
        fj = df.loc[(df[0] == e) & (df[1] == propm)]
        fk = df.loc[(df[0] == e) & (df[1] == propn)]

        if not fj.empty and not fk.empty:
            similarity = sim(
                fj[2].str.lower().values[0],
                fk[2].str.lower().values[0],
                p=0.6,
                l=0.1,
            )
            if similarity >= theta and similarity <= 1:
                supp_plus += 1
    return supp_plus


def extract_uri_or_literal(value, last_seg: bool):
    match = re.match(r"<(.*?)>", value)
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


def parse_triple(triple: str) -> tuple[str, str, str]:
    triple = triple.strip(" .")  # Remove trailing space and dot
    subject, predicate, obj = triple.split(" ", 2)  # Split into three parts
    return subject.strip("<>"), predicate.strip("<>"), obj.strip("<>")


def convert_triples_to_dict(
    triples: list[str],
) -> dict[str, list[tuple[str, str, str]]]:  # convert the triple into dictionary format
    result = {}
    for triple in triples:
        subject, predicate, obj = parse_triple(triple)
        if subject not in result:
            result[subject] = []
        result[subject].append((subject, predicate, obj))
    return result


if __name__ == "__main__":

    filtered_1_gram_path = "filtered_1_gram.csv"
    filtered_2_gram_path = "filtered_2_gram.csv"

    df = pd.read_csv("dbpedia.csv", sep=",", header=None)

    df_class = pd.read_csv("output_classes.csv")  # superclass, class columns
    df_prop = pd.read_csv("output_property.csv")  # superprop, prop columns

    nt_file_path = "matched_entities_triples.nt"

    unique_prop = df[1].unique()
    unique_entity = df[0].unique()
    unique_prop = unique_prop.tolist()
    unique_entity = unique_entity.tolist()
    num_unique_prop = len(unique_prop)
    unique_prop_value_pairs = df[[1, 2]].drop_duplicates().apply(tuple, axis=1).tolist()
    num_unique_prop_value_pairs = len(unique_prop_value_pairs)
    prop_to_index = {prop: idx for idx, prop in enumerate(unique_prop)}
    prop_value_to_index = {
        pair: idx for idx, pair in enumerate(unique_prop_value_pairs)
    }

    prop_min_page_count = [0] * num_unique_prop
    prop_min_vol_count = [0] * num_unique_prop
    utility_prop = [0] * num_unique_prop

    rows, cols = (num_unique_prop, num_unique_prop)
    supp = [[0 for i in range(cols)] for j in range(rows)]
    supp_plus = [[0 for i in range(cols)] for j in range(rows)]
    confidence = [[0 for i in range(cols)] for j in range(rows)]

    # # # print(unique_values_list)
    for index, prop in enumerate(unique_prop):
        min_page_count, min_vol_count = calc_n_gram_vols(
            filtered_1_gram_path, filtered_2_gram_path, prop
        )
        # print(min_page_count,min_vol_count)
        prop_min_page_count[index] = min_page_count
        prop_min_vol_count[index] = min_vol_count

        utility_prop[index] = expectation(min_vol_count, k, B) * logP(
            df, prop, num_unique_prop
        )

    for i in range(len(unique_prop)):
        for j in range(i + 1, len(unique_prop)):
            propm = unique_prop[i]
            propn = unique_prop[j]
            supp[i][j] = support(propm, propn, df)
            supp_plus[i][j] = support_plus(propm, propn, df, 0.7)
            if supp_plus[i][j] == 0 or supp[i][j] == 0:
                confidence[i][j] = 0
            else:
                confidence[i][j] = supp_plus[i][j] / supp[i][j]

    class_hierarchy = build_hierarchy_map(
        df_class, "last_word_superclass", "last_word_subclass"
    )
    prop_hierarchy = build_hierarchy_map(
        df_prop, "last_word_superclass", "last_word_subclass"
    )
    rho_matrices = {}
    for entity in unique_entity:
        # Filter rows for the current entity
        entity_df = df[df[0] == entity].reset_index(drop=True)
        rho = np.zeros(
            (num_unique_prop_value_pairs, num_unique_prop_value_pairs), dtype=int
        )

        # Populate the rho matrix based on conditions
        for j in range(len(entity_df)):
            for k in range(len(entity_df)):
                # Extract properties and values
                prop_j, val_j = entity_df.loc[j, 1], entity_df.loc[j, 2]
                prop_k, val_k = entity_df.loc[k, 1], entity_df.loc[k, 2]

                index_j = prop_value_to_index[(prop_j, val_j)]
                index_k = prop_value_to_index[(prop_k, val_k)]

                # Check if both properties are 'type' and the related values are subclasses or super classes
                if (
                    prop_j == "type"
                    and prop_k == "type"
                    and is_related_value(val_j, val_k, class_hierarchy)
                ):
                    rho[index_j][index_k] = 1
                # Check if both values are same and the related values are subclasses or super classes
                if val_j == val_k and is_related_property(
                    prop_j, prop_k, prop_hierarchy
                ):
                    rho[index_j][index_k] = 1
        for i in range(len(unique_prop)):
            for j in range(i + 1, len(unique_prop)):
                if confidence[i][j] > 0.91 and supp[i][j] > 200:
                    prop_i = unique_prop[i]
                    prop_j = unique_prop[j]
                    indices_i = [
                        idx
                        for idx, pair in enumerate(unique_prop_value_pairs)
                        if pair[0] == prop_i
                    ]
                    indices_j = [
                        idx
                        for idx, pair in enumerate(unique_prop_value_pairs)
                        if pair[0] == prop_j
                    ]
                    for idx_i in indices_i:
                        for idx_j in indices_j:
                            rho[idx_i][idx_j] = 1
        # Store the rho matrix for the current entity
        rho_matrices[entity] = rho

    # objective_coefficients = utility_prop
    # knapsack_problem = pulp.LpProblem("Knapsack_with_Dependencies", pulp.LpMaximize)
    # unique_entity = unique_entity[:1]
    entities = {}
    results = {}
    L = 10
    lengths_prop = [1 for i in range(num_unique_prop_value_pairs)]
    for entity in unique_entity:
        entity_df = df[df[0] == entity].reset_index(drop=True)
        entity_prop_values = (
            entity_df[[1, 2]].drop_duplicates().apply(tuple, axis=1).tolist()
        )
        entities[entity] = [prop_value_to_index[pair] for pair in entity_prop_values]

    matched_triples = []

    for entity in unique_entity:
        knapsack_problem = pulp.LpProblem(
            f"Knapsack_with_Dependencies_Entity_{entity}", pulp.LpMaximize
        )
        props = entities[entity]
        rho = rho_matrices[entity]

        X = {i: pulp.LpVariable(f"X_{i}", cat="Binary") for i in props}
        knapsack_problem += (
            pulp.lpSum(
                [
                    utility_prop[prop_to_index[unique_prop_value_pairs[i][0]]] * X[i]
                    for i in props
                ]
            ),
            "Total_Utility",
        )
        knapsack_problem += (
            pulp.lpSum([lengths_prop[i] * X[i] for i in props]) <= L,
            f"Length_Constraint_Entity_{entity}",
        )
        for j in props:
            for k in props:
                if j < k and rho[j][k] == 1:
                    knapsack_problem += (
                        X[j] + X[k] <= 1,
                        f"Dependency_Constraint_{j}_{k}_Entity_{entity}",
                    )
        knapsack_problem.solve()
        selected_items = [i for i in props if X[i].value() == 1]
        selected_item_names = [
            unique_prop_value_pairs[i] for i in props if X[i].value() == 1
        ]
        max_value = pulp.value(knapsack_problem.objective)
        results[entity] = {
            "max_value": max_value,
            "selected_items": selected_item_names,
        }
        # print(entity, results[entity])
        predicate_object_pairs = results[entity]["selected_items"]
        with open(nt_file_path, "r") as ntfile:
            for line in ntfile:
                if line.strip():  # Skip empty lines
                    # Extract parts of the triple
                    match = re.match(r"(<.*?>) (<.*?>) (.*) \.", line)
                    if match:
                        subject_uri, predicate_uri, object_value = match.groups()
                        # Extract URIs or literals from the triple
                        subject = extract_uri_or_literal(
                            subject_uri, True
                        )  # extract last part of uri
                        predicate = extract_uri_or_literal(
                            predicate_uri, True
                        )  # extract last part of uri
                        object = extract_uri_or_literal(object_value, False)
                        # if '"' in object_value:
                        # print(subject_uri, object_value, object)
                        # print(subject, predicate, object)

                        # Check for matches
                        for (
                            expected_predicate,
                            expected_object,
                        ) in predicate_object_pairs:
                            if (
                                predicate == expected_predicate
                                and object == expected_object
                            ):
                                matched_triples.append(
                                    f"{subject_uri} {predicate_uri} {object}"
                                )
    # print(matched_triples)
    converted_data = convert_triples_to_dict(matched_triples)
    toolkit = WikESToolkit()
    G = toolkit.load_graph(
        ESBMGraph,
        ESBMVersions.V1Dot2.DBPEDIA_FULL,  # ESBMVersions.Plus.DBPEDIA_FULL or ESBMVersions.V1Dot2.DBPEDIA_TEST_0 or ESBMVersions.V1Dot2.LMDB_TRAIN_1
        entity_formatter=lambda e: e.identifier,
    )
    root_nodes = G.root_entities()
    first_root_node = G.root_entity_ids()[0]
    nodes = G.entities()
    edges = G.triples()
    labels = G.predicates()
    number_of_nodes = G.total_entities()
    number_of_directed_edges = G.total_triples()
    for root_entity, triples in converted_data.items():
        # print(triples)
        G.mark_triples_as_summaries(root_entity, triples)

    f1_10 = G.f1_score(10)
    print("f1 score for 10 summaries", f1_10)
