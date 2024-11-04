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

B = 4541627  # total number of books
total_number_of_pages = 2441898561
eps = 10**-6


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


def familiarityFunction(k):
    if B == 0 or k == 0:
        print("error")
        return 0
    return math.log(k + 1) / math.log(B + 1)


def expectation(b_f, M):
    Max = min(b_f, M)
    Min, exppectation, factor = 1, 0, 0
    common_factor = (
        math.lgamma(b_f + 1)
        + math.lgamma(B - b_f)
        + math.lgamma(M)
        + math.lgamma(B - M + 1)
        - math.lgamma(M)
    )
    for k in range(Min, Max + 1):
        factor = (
            math.lgamma(k)
            + math.lgamma(M - k)
            + math.lgamma(b_f - k + 1)
            + math.lgamma(B - M - b_f + k)
        )
        factor = common_factor - factor
        expectation += math.exp(factor) * familiarityFunction(k)
    return expectation


def logP(df, prop, num_unique_prop):
    filtered_df = df[df[1] == prop]
    filtered_df = filtered_df[0].unique()
    prob_feature = (-1) * math.log(len(filtered_df) / (num_unique_prop + eps))
    return prob_feature


def calc_n_gram_vols(url_list_2_gram, url_list_1_gram, prop):
    filename1 = "one_gram_file"
    filename2 = "two_gram_file"
    one_grams = []
    two_grams = []
    min_page_count = float("inf")
    min_vol_count = float("inf")
    split_words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", prop)
    one_grams.extend(split_words)
    two_grams.extend(
        [" ".join(split_words[i : i + 2]) for i in range(len(split_words) - 1)]
    )
    print(one_grams, two_grams)
    filtered_1_gram = pd.read_csv("filtered_1_gram", sep=r"[ \t]+", header=None)
    for one_gram in one_grams:
        page_count = 0
        vol_count = 0
        filtered_df = filtered_1_gram[
            filtered_1_gram[0].astype(str).str.lower() == one_gram.lower()
        ]
        page_count += filtered_df[2].sum()
        vol_count += filtered_df[3].sum()
        if page_count < min_page_count:
            min_page_count = page_count
        if vol_count < min_vol_count:
            min_vol_count = vol_count

    for two_gram in two_grams:

        first_word, second_word = two_gram.split()
        page_count = 0
        vol_count = 0
        filtered_2_gram = pd.read_csv(
            "filtered_output_2gram", sep=r"[ \t]+", header=None
        )
        filtered_df = filtered_2_gram[
            (filtered_2_gram[0].astype(str).str.lower() == first_word.lower())
            & (filtered_2_gram[1].astype(str).str.lower() == second_word.lower())
        ]
        page_count += filtered_df[3].sum()
        vol_count += filtered_df[4].sum()

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
    print(two_grams_lower)

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
    print(intersection)
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


if __name__ == "__main__":

    with open("1gram.txt", "r") as file:
        url_list_1_gram = file.read().splitlines()

    with open("2gram.txt", "r") as file:
        url_list_2_gram = file.read().splitlines()

    df = pd.read_csv("dbpedia.csv", sep=",", header=None)

    df_class = pd.read_csv("output_classes.csv")  # superclass, class columns
    df_prop = pd.read_csv("output_property.csv")  # superprop, prop columns

    unique_prop = df[1].unique()
    unique_entity = df[0].unique()
    unique_prop = unique_prop.tolist()
    unique_entity = unique_entity.tolist()
    num_unique_prop = len(unique_prop)
    prop_to_index = {prop: idx for idx, prop in enumerate(unique_prop)}

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
            url_list_2_gram, url_list_1_gram, prop
        )
        # print(min_page_count,min_vol_count)
        prop_min_page_count[index] = min_page_count
        prop_min_vol_count[index] = min_vol_count
        utility_prop[index] = expectation(min_vol_count, 10) * logP(
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
        rho = np.zeros((num_unique_prop, num_unique_prop), dtype=int)

        # Populate the rho matrix based on conditions
        for j in range(len(entity_df)):
            for k in range(len(entity_df)):
                # Extract properties and values
                prop_j, val_j = entity_df.loc[j, 1], entity_df.loc[j, 2]
                prop_k, val_k = entity_df.loc[k, 1], entity_df.loc[k, 2]

                # Check if both properties are 'type' and the related values are subclasses or super classes
                if (
                    prop_j == "type"
                    and prop_k == "type"
                    and is_related_value(val_j, val_k, class_hierarchy)
                ):
                    rho[j][k] = 1
                # Check if both values are same and the related values are subclasses or super classes
                if val_j == val_k and is_related_property(
                    prop_j, prop_k, prop_hierarchy
                ):
                    rho[j][k] = 1
        for i in range(len(unique_prop)):
            for j in range(i + 1, len(unique_prop)):
                if confidence[i][j] > 0.91 and supp[i][j] > 200:
                    rho[i][j] = 1
        # Store the rho matrix for the current entity
        rho_matrices[entity] = rho

    objective_coefficients = utility_prop
    knapsack_problem = pulp.LpProblem("Knapsack_with_Dependencies", pulp.LpMaximize)
