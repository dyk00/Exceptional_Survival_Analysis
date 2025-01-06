import sys
import os
from data_processing.data_io import read_parquet

df, attributes = read_parquet("example_data.parquet", "./data")
print(df.head())
