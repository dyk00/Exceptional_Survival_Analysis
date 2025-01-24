import os
import pandas as pd


def read_parquet(folder, name):
    path = os.path.join(folder, name)
    df = pd.read_parquet(path, engine="pyarrow")
    cols = df.columns.to_list()
    return df, cols


def save_parquet(df, folder, name):
    path = os.path.join(folder, name)
    df.to_parquet(path, engine="pyarrow", index=False)


# # test code
# folder = "./data"
# name = "example_data.parquet"
# df, cols = read_parquet(folder, name)
# print(df.head())
# print(df.dtypes)
# save_parquet(df, "./data", "new_example_data.parquet")
