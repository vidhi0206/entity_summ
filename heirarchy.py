import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("dbpedia.csv", sep=",", header=None)  # entity, prop, val columns
df_class = pd.read_csv("output_classes.csv")  # superclass, class columns
df_prop = pd.read_csv("output_property.csv")  # superprop, prop columns


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
