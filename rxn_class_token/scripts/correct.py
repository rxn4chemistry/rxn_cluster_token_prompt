"""
This script is used because we found an error in the data: multiple reactions with different product and same set of precursors
"""

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

datapath = '/Users/ato/Library/CloudStorage/Box-Box/IBM RXN for Chemistry/Data/class_token/std_pistachio_201002'
train_df = pd.read_csv(f"{datapath}/df.with-reagents.train.csv")
train_df['split'] = "train"

valid_df = pd.read_csv(f"{datapath}/df.with-reagents.valid.csv")
valid_df['split'] = "valid"

test_df = pd.read_csv(f"{datapath}/df.with-reagents.test.csv")
test_df['split'] = "test"

df = pd.concat([train_df, valid_df, test_df])
df['precursors'] = df["rxn"].progress_apply(lambda x: x.split('>>')[0].strip())
print(df.columns)

# This is a cleaning needed to remove reactions with different product but equal precursors
print(f"Length: {len(df)}")
# create a list of unique precursors
unique_prec = set(df.precursors.values)
dict_unique_prec = {}
for i, elem in enumerate(unique_prec):
    dict_unique_prec[elem] = i

# give the same index to the repetitions
df['repeated'] = df.precursors.progress_apply(lambda x: dict_unique_prec[x])

df_repeated = df[df.duplicated(['repeated'], keep=False)].sort_values('repeated').reset_index()
repeated_indices = list(set(df_repeated.repeated.values))

# cleaning
df['to_remove'] = df.progress_apply(lambda x: (x['repeated'] in repeated_indices), axis=1)
df = df[~df.to_remove].reset_index()
print(f"Length: {len(df)}")

df[['rxn', 'rxnclasses', 'classes']].to_csv(f"{datapath}/corrected.csv")
