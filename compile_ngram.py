import os
import requests
import gzip
import pandas as pd
import math
import re

B = 4541627 # total number of books
total_number_of_pages=2441898561

def download_file(url, filename):
    # Download the file
    print(f"Downloading file from {url}...")
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        with open(filename, 'wb') as file:
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
    with gzip.open(file_path, 'rt') as gz_file:
        df = pd.read_csv(gz_file, sep=r'[ \t]+', engine='python',header=None)
    return df

def familiarityFunction(k):
    if B == 0 or k==0:
        print("error")
        return 0
    return math.log(k+1)/math.log(B+1)

def expectation(b_f, M):
    Max = min(b_f, M)
    Min, exppectation ,factor = 1, 0, 0
    common_factor = math.lgamma(b_f+1) + math.lgamma(B-b_f) + math.lgamma(M) + math.lgamma(B-M+1)-math.lgamma(M)
    for k in range(Min, Max+1):
        factor = math.lgamma(k)+math.lgamma(M-k)+math.lgamma(b_f-k+1)+math.lgamma(B-M-b_f+k)
        factor = common_factor-factor
        expectation += math.exp(factor)*familiarityFunction(k)
    return expectation


def calc_n_gram_vols(url_list_2_gram,url_list_1_gram, prop):
    filename1 = "one_gram_file"
    filename2 = "two_gram_file"
    one_grams = []
    two_grams = []
    min_page_count = float('inf')
    min_vol_count = float('inf')
    split_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', prop)
    one_grams.extend(split_words)
    two_grams.extend([' '.join(split_words[i:i+2]) for i in range(len(split_words)-1)])
    print(one_grams,two_grams)
    # for one_gram in one_grams:
    #     page_count = 0
    #     vol_count = 0
    #     for url in url_list_1_gram:
    #         download_file(url,filename1)
    #         df = read_gz_csv(filename1)
    #         filtered_df = df[df[0].astype(str).str.lower() == one_gram.lower()]
    #         page_count += filtered_df[2].sum()
    #         vol_count += filtered_df[3].sum()
    #         print(filtered_df[3].sum())
    #         delete_file(filename1)
    #     if page_count < min_page_count:
    #         min_page_count = page_count
    #     if vol_count < min_vol_count:
    #         min_vol_count = vol_count
    for two_gram in two_grams:
        
        first_word, second_word = two_gram.split()
        page_count = 0
        vol_count = 0
        for url in url_list_2_gram:
            download_file(url,filename2)
            #df = read_gz_csv(filename2)
            with gzip.open(filename2, 'rt') as gz_file:
                for df in pd.read_csv(gz_file,  sep=r'[ \t]+', engine='python',header=None, chunksize=100000):
                    filtered_df = df[(df[0].astype(str).str.lower() == first_word.lower()) & (df[1].astype(str).str.lower() == second_word.lower())]
                    page_count += filtered_df[3].sum()
                    vol_count += filtered_df[4].sum()
            delete_file(filename2)
        print(vol_count)
        if page_count < min_page_count:
            min_page_count = page_count
        if vol_count < min_vol_count:
            min_vol_count = vol_count
    return min_page_count,min_vol_count

def filter_df(unique_prop,url_list_2_gram,url_list_1_gram):
    filename1 = "googlebooks-eng-all-1-1gram-20120701"
    filename2 = "googlebooks-eng-all-2-1gram-20120701"
    one_grams = []
    two_grams = []
    for prop in unique_prop:
        split_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', prop)
        one_grams.extend(split_words)
        two_grams.extend([' '.join(split_words[i:i+2]) for i in range(len(split_words)-1)])

    #print(one_grams,two_grams)
    # filtered_df_1_gram = pd.DataFrame()
    # for url in url_list_1_gram:
    #     download_file(url,filename1)
    #     with gzip.open(filename1, 'rt') as gz_file:
    #         for df in pd.read_csv(gz_file,  sep=r'[ \t]+', engine='python',header=None, chunksize=100000):
    #             for one_gram in one_grams:
    #                 current_filtered_df = df[df[0].astype(str).str.lower() == one_gram.lower()]
    #                 filtered_df_1_gram = pd.concat([filtered_df_1_gram, current_filtered_df])
    #     delete_file(filename1)
    # filtered_df_1_gram.to_csv("filtered_1_gram.csv", sep='\t', encoding='utf-8', index=False)
    filtered_df_1_gram = pd.DataFrame()
    for url in url_list_2_gram:
        download_file(url,filename2)
        with gzip.open(filename2, 'rt') as gz_file:
            for df in pd.read_csv(gz_file,  sep=r'[ \t]+', engine='python',header=None, chunksize=1000000):
                for two_gram in two_grams:
                    first_word, second_word = two_gram.split()
                    current_filtered_df = df[(df[0].astype(str).str.lower() == first_word.lower()) & (df[1].astype(str).str.lower() == second_word.lower())]
                    filtered_df_1_gram = pd.concat([filtered_df_1_gram, current_filtered_df])
        delete_file(filename2)
    
    
    filtered_df_1_gram.to_csv("filtered_2_gram.csv", sep='\t', encoding='utf-8', index=False)
    
if __name__=="__main__":
    # Example usage
    #filename = "googlebooks-eng-all-2gram-20120701-0.gz"

    with open('1gram.txt', 'r') as file:
        url_list_1_gram = file.read().splitlines()

    with open('2gram.txt', 'r') as file:
        url_list_2_gram = file.read().splitlines()

    df = pd.read_csv("dbpedia.csv", sep=',',header=None)
    # # print(url_list_1_gram[0:5])
    # # Load names from entities.txt
    # # with open('entities.txt', 'r') as file:
    # #     entity_names = file.read().splitlines()

    # # dbpedia_df = df1[df1[0].isin(entity_names)]

    # # dbpedia_df.to_csv('dbpedia.csv', index=False, header=False)

    unique_prop = df[1].unique()
    unique_prop = unique_prop.tolist()
    num_unique_prop = len(unique_prop)

    prop_min_page_count = [0] * num_unique_prop
    prop_min_vol_count = [0] * num_unique_prop
    utility_prop = [0] * num_unique_prop
    filter_df(unique_prop, url_list_2_gram, url_list_1_gram)

    # # # print(unique_values_list)
    # unique_values_list = ["homeAlone"]
    # for index , prop in enumerate(unique_values_list):
    #     #min_page_count,min_vol_count = calc_n_gram_vols(url_list_2_gram,url_list_1_gram,prop)
    #     #print(min_page_count,min_vol_count)
    #     # prop_min_page_count[index] = min_page_count
    #     # prop_min_vol_count[index] = min_vol_count
    #     # utility_prop[index] = expectation(min_vol_count,10)
        

    