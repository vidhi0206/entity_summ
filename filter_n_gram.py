import os
import requests
import gzip
import re
import pandas as pd


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


def filter_df_1gram(unique_prop, url_list_1_gram):
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


if __name__ == "__main__":
    df = pd.read_csv("dbpedia.csv", sep=",", header=None)
    unique_prop = df[1].unique()
    unique_prop = unique_prop.tolist()
    num_unique_prop = len(unique_prop)
    filter_df_1gram(unique_prop, "1gram.txt")
    filter_df_2gram(unique_prop, "2gram.txt")
