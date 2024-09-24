import pandas as pd

# Define the file path and delimiter (e.g., ',' for comma or '\t' for tab)
file_path = 'elist.txt'
delimiter = '\t'  # Change this if your file uses a different delimiter

# Read the file into a pandas DataFrame
df = pd.read_csv(file_path, delimiter=delimiter)
df = df[df["dataset"]== "dbpedia"]
df['prop_name'] = df['euri'].str.split('/').str[-1]
df['prop_name'].to_csv('entities.txt', index=False)
print(df)