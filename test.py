import pandas as pd
import math
    
df = pd.read_csv("dbpedia.csv", sep=',',header=None)

unique_prop = df[1].unique()
unique_prop = unique_prop.tolist()
num_unique_prop = len(unique_prop)

prob_feature  = [0] * num_unique_prop
eps = 10**-6

for index, prop in enumerate(unique_prop):
    filtered_df=df[df[1]==prop]
    filtered_df=filtered_df[0].unique()
    prob_feature[index] = (-1)*math.log(len(filtered_df)/(num_unique_prop + eps))
    print(prop,len(filtered_df),prob_feature[index])