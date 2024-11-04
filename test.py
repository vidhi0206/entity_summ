# import pandas as pd
# import math

# df = pd.read_csv("dbpedia.csv", sep=',',header=None)

# unique_prop = df[1].unique()
# unique_prop = unique_prop.tolist()
# num_unique_prop = len(unique_prop)

# prob_feature  = [0] * num_unique_prop
# eps = 10**-6

# for index, prop in enumerate(unique_prop):
#     filtered_df=df[df[1]==prop]
#     filtered_df=filtered_df[0].unique()
#     prob_feature[index] = (-1)*math.log(len(filtered_df)/(num_unique_prop + eps))
#     print(prop,len(filtered_df),prob_feature[index])

# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("property.csv", sep=",")  # Replace with your file path


# # Function to extract the last word from a URL
# def extract_last_word(url):
#     return url.rstrip("/").split("/")[-1]


# # Apply the function to both columns
# df["last_word_superclass"] = df.iloc[:, 0].apply(extract_last_word)
# df["last_word_subclass"] = df.iloc[:, 1].apply(extract_last_word)

# # Save the updated dataframe to a new CSV file
# df.to_csv("output_property.csv", index=False)

# print("Extraction complete. Saved as 'output_file.csv'.")
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("dbpedia.csv", sep=",", header=None)  # entity, prop, val columns
df_class = pd.read_csv("output_classes.csv")  # superclass, class columns
df_prop = pd.read_csv("output_property.csv")  # superprop, prop columns
print(df.head())


# Helper function to build superclass and superproperty maps
def build_hierarchy_map(df_hierarchy, super_col, sub_col):
    hierarchy_map = df_hierarchy.groupby(sub_col)[super_col].apply(list).to_dict()

    # Recursive function to find all ancestors
    def find_all_super_relations(item, hierarchy, visited=None):
        if visited is None:
            visited = set()
        if item not in hierarchy or item in visited:
            return set()
        visited.add(item)
        ancestors = set(hierarchy[item])
        for ancestor in hierarchy[item]:
            ancestors.update(find_all_super_relations(ancestor, hierarchy, visited))
        return ancestors

    # Expand the map to include indirect superclasses and superproperties
    full_hierarchy = {
        item: find_all_super_relations(item, hierarchy_map) for item in hierarchy_map
    }
    return full_hierarchy


# Build subclass and subproperty relations
class_hierarchy = build_hierarchy_map(
    df_class, "last_word_superclass", "last_word_subclass"
)
prop_hierarchy = build_hierarchy_map(
    df_prop, "last_word_superclass", "last_word_subclass"
)


# Function to check if prop_a is related to prop_b
def is_related_property(prop_a, prop_b, hierarchy):
    return (
        prop_a == prop_b
        or prop_b in hierarchy.get(prop_a, set())
        or prop_a in hierarchy.get(prop_b, set())
    )


# Function to check if val_a is related to val_b
def is_related_value(val_a, val_b, hierarchy):
    return (
        val_a == val_b
        or val_b in hierarchy.get(val_a, set())
        or val_a in hierarchy.get(val_b, set())
    )


# Generate rho matrices for each entity
rho_matrices = {}
unique_entity = df[0].unique()
unique_prop = df[1].unique()
n = len(unique_prop)
prop_to_index = {prop: idx for idx, prop in enumerate(unique_prop)}
for entity in unique_entity:
    print(n)
    # Filter rows for the current entity
    entity_df = df[df[0] == entity].reset_index(drop=True)
    rho = np.zeros((n, n), dtype=int)

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
            if val_j == val_k and is_related_property(prop_j, prop_k, prop_hierarchy):
                rho[j][k] = 1

    # Store the rho matrix for the current entity
    rho_matrices[entity] = rho

# Output results
for entity, rho in rho_matrices.items():
    print(f"Entity: {entity}\nRho matrix:\n{rho}\n")
