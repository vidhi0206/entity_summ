# import pandas as pd

# # Define the file path and delimiter (e.g., ',' for comma or '\t' for tab)
# file_path = 'elist.txt'
# delimiter = '\t'  # Change this if your file uses a different delimiter

# # Read the file into a pandas DataFrame
# df = pd.read_csv(file_path, delimiter=delimiter)
# df = df[df["dataset"]== "dbpedia"]
# df['prop_name'] = df['euri'].str.split('/').str[-1]
# df['prop_name'].to_csv('entities.txt', index=False)
# print(df)

import pandas as pd
import re

# File paths
nt_file_path = "complete_data/dbpedia/complete_dbpedia.nt"
elist_file_path = "elist.txt"
output_file_path = "matched_entities_triples.nt"

# Load elist.txt using pandas
elist_df = pd.read_csv(elist_file_path, sep="\t")

# Extract euri column into a set for fast lookup
euri_set = set(elist_df["euri"])


# Helper function to extract the URI without angle brackets
def extract_uri(line):
    match = re.match(r"<(.*?)>", line)
    return match.group(1) if match else None


# Parse the .nt file and filter matching triples
matched_triples = []
with open(nt_file_path, "r") as ntfile:
    for line in ntfile:
        if line.strip():  # Skip empty lines
            # Extract parts of the triple
            match = re.match(r"(<.*?>) (<.*?>) (.*?) \.", line)
            if match:
                subject, predicate, obj = match.groups()

                # Check if the subject URI is in euri_set
                if extract_uri(subject) in euri_set:
                    matched_triples.append(line.strip())

# Save matched triples to output file
with open(output_file_path, "w") as output_file:
    output_file.write("\n".join(matched_triples))

# Print a sample of the matched triples
print("Sample Matched Triples:")
for triple in matched_triples[:5]:
    print(triple)
