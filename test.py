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
# import pandas as pd
# import numpy as np

# # Load the data
# df = pd.read_csv("dbpedia.csv", sep=",", header=None)  # entity, prop, val columns
# df_class = pd.read_csv("output_classes.csv")  # superclass, class columns
# df_prop = pd.read_csv("output_property.csv")  # superprop, prop columns
# print(df.head())


# # Helper function to build superclass and superproperty maps
# def build_hierarchy_map(df_hierarchy, super_col, sub_col):
#     hierarchy_map = df_hierarchy.groupby(sub_col)[super_col].apply(list).to_dict()

#     # Recursive function to find all ancestors
#     def find_all_super_relations(item, hierarchy, visited=None):
#         if visited is None:
#             visited = set()
#         if item not in hierarchy or item in visited:
#             return set()
#         visited.add(item)
#         ancestors = set(hierarchy[item])
#         for ancestor in hierarchy[item]:
#             ancestors.update(find_all_super_relations(ancestor, hierarchy, visited))
#         return ancestors

#     # Expand the map to include indirect superclasses and superproperties
#     full_hierarchy = {
#         item: find_all_super_relations(item, hierarchy_map) for item in hierarchy_map
#     }
#     return full_hierarchy


# # Build subclass and subproperty relations
# class_hierarchy = build_hierarchy_map(
#     df_class, "last_word_superclass", "last_word_subclass"
# )
# prop_hierarchy = build_hierarchy_map(
#     df_prop, "last_word_superclass", "last_word_subclass"
# )


# # Function to check if prop_a is related to prop_b
# def is_related_property(prop_a, prop_b, hierarchy):
#     return (
#         prop_a == prop_b
#         or prop_b in hierarchy.get(prop_a, set())
#         or prop_a in hierarchy.get(prop_b, set())
#     )


# # Function to check if val_a is related to val_b
# def is_related_value(val_a, val_b, hierarchy):
#     return (
#         val_a == val_b
#         or val_b in hierarchy.get(val_a, set())
#         or val_a in hierarchy.get(val_b, set())
#     )


# # Generate rho matrices for each entity
# rho_matrices = {}
# unique_entity = df[0].unique()
# unique_prop = df[1].unique()
# n = len(unique_prop)
# prop_to_index = {prop: idx for idx, prop in enumerate(unique_prop)}
# for entity in unique_entity:
#     print(n)
#     # Filter rows for the current entity
#     entity_df = df[df[0] == entity].reset_index(drop=True)
#     rho = np.zeros((n, n), dtype=int)

#     # Populate the rho matrix based on conditions
#     for j in range(len(entity_df)):
#         for k in range(len(entity_df)):
#             # Extract properties and values
#             prop_j, val_j = entity_df.loc[j, 1], entity_df.loc[j, 2]
#             prop_k, val_k = entity_df.loc[k, 1], entity_df.loc[k, 2]

#             # Check if both properties are 'type' and the related values are subclasses or super classes
#             if (
#                 prop_j == "type"
#                 and prop_k == "type"
#                 and is_related_value(val_j, val_k, class_hierarchy)
#             ):
#                 rho[j][k] = 1
#             # Check if both values are same and the related values are subclasses or super classes
#             if val_j == val_k and is_related_property(prop_j, prop_k, prop_hierarchy):
#                 rho[j][k] = 1

#     # Store the rho matrix for the current entity
#     rho_matrices[entity] = rho

# # Output results
# for entity, rho in rho_matrices.items():
#     print(f"Entity: {entity}\nRho matrix:\n{rho}\n")

import pulp
import numpy as np

num_entities = 2  # Total number of entities
num_unique_props = 5  # Total unique properties across all entities
L = 2  # Maximum allowed length for each entity

# Sample data
prop_to_index = {
    f"prop_{i}": i for i in range(num_unique_props)
}  # Property to index mapping
utility_prop = np.random.rand(
    num_unique_props
)  # Utility for each property: q * (-log p)
lengths_prop = np.random.randint(
    1, 10, size=num_unique_props
)  # Length for each property

# Sample entity data
# For each entity, we have a subset of properties and its dependency rho matrix
entities = {}
rho_matrices = {}

for entity_id in range(num_entities):
    # Sample subset of properties for this entity
    entity_props = np.random.choice(
        list(prop_to_index.keys()), size=np.random.randint(1, 5), replace=False
    )
    print(entity_props)
    entities[entity_id] = [
        prop_to_index[prop] for prop in entity_props
    ]  # Store property indices for the entity
    print(entities[entity_id])
    # Sample rho matrix for this entity
    rho = np.zeros((num_unique_props, num_unique_props))
    for j in entities[entity_id]:
        for k in entities[entity_id]:
            if j < k:
                rho[j][k] = np.random.choice(
                    [0, 1]
                )  # Randomly set dependencies for illustration
    rho_matrices[entity_id] = rho

# Solve knapsack problem for each entity
results = {}
for entity_id in range(num_entities):
    # Initialize the problem for this entity
    knapsack_problem = pulp.LpProblem(
        f"Knapsack_with_Dependencies_Entity_{entity_id}", pulp.LpMaximize
    )

    # Extract relevant properties for the entity
    props = entities[entity_id]
    rho = rho_matrices[entity_id]

    # Define binary decision variables for this entity
    X = {i: pulp.LpVariable(f"X_{i}", cat="Binary") for i in props}

    # Objective function: Maximize sum of (X_i * utility_prop[i])
    knapsack_problem += (
        pulp.lpSum([utility_prop[i] * X[i] for i in props]),
        "Total_Utility",
    )

    # Length constraint for the entity
    knapsack_problem += (
        pulp.lpSum([lengths_prop[i] * X[i] for i in props]) <= L,
        f"Length_Constraint_Entity_{entity_id}",
    )

    # Pairwise dependency constraints
    for j in props:
        for k in props:
            if j < k and rho[j][k] == 1:
                knapsack_problem += (
                    X[j] + X[k] <= 1,
                    f"Dependency_Constraint_{j}_{k}_Entity_{entity_id}",
                )

    # Solve the problem
    knapsack_problem.solve()

    # Store the result
    selected_items = [i for i in props if X[i].value() == 1]
    max_value = pulp.value(knapsack_problem.objective)
    results[entity_id] = {"max_value": max_value, "selected_items": selected_items}

# Display results
# for entity_id, result in results.items():
#     print(f"Entity {entity_id}:")
#     print(f"  Maximum objective function value: {result['max_value']}")
#     print(f"  Selected properties: {result['selected_items']}")
