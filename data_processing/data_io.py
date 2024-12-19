import os
import pandas as pd


def read_parquet(name, folder):
    path = os.path.join(folder, name)
    df = pd.read_parquet(path, engine="pyarrow")
    attributes = df.columns.to_list()
    return df, attributes


def save_parquet(df, name, folder):
    path = os.path.join(folder, name)
    df.to_parquet(path, engine="pyarrow", index=False)


# # test code
# folder = "./data"
# name = "example_data.parquet"
# df, attributes = read_parquet(name, folder)
# print(df.head())
# print(df.dtypes)
# save_parquet(df, "new_example_data", "./data")
