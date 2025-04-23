import math
import pandas as pd
from .subgroup import filter_data

nominal_vars = [
    "specialisme_code",
    "first_event",
    "count_death_fullcode",
    "count_death_ic",
    "count_ic_6hr",
    "count_acute_ic",
    "hoofdverrichting_code",
    "prioriteit_code",
    "care_order",
    "m_year",
    "m_month",
    "m_day",
    "m_hour",
]

# add new descriptions to the seed description
def refinement_operator(seed_desc, dataset, columns, b, level):

    # initialize new_descriptions
    new_descriptions = []

    # get the current subgroup
    current_subgroup = filter_data(seed_desc, dataset)

    # existing descriptions
    current_descriptions = set(desc[0] for desc in seed_desc)

    for col in columns:
        # skip if used
        if col in current_descriptions:
            continue

        # subgroup_cols = current_subgroup[col].dropna()
        subgroup_cols = current_subgroup[col]

        if len(subgroup_cols) == 0:
            continue

        # skip unhashable errors in case inputting wrong vars
        try:
            unique_vals = subgroup_cols.unique()
        except TypeError:
            # print(f"Depth {level} Skipping {col}")
            continue

        # for bools or only 2 values
        if pd.api.types.is_bool_dtype(subgroup_cols) or len(unique_vals) == 2:
            bin_values = unique_vals
            for val in bin_values:
                desc_new = seed_desc + [(col, ("=", val))]
                new_descriptions.append(desc_new)
            # print(f"Depth {level} Processing {col}. Values = {bin_values}")

        # for nominal variables that are specified
        # (otherwise it'll be processed as numerical vars)
        elif col in nominal_vars:
            nominal_values = unique_vals
            for val in nominal_values:
                desc_eq = seed_desc + [(col, ("=", val))]
                desc_neq = seed_desc + [(col, ("!=", val))]
                new_descriptions.append(desc_eq)
                new_descriptions.append(desc_neq)
            # print(f"Depth {level} Processing {col}. Values = {nominal_values}")

        # for nominal variables string or category types
        elif pd.api.types.is_string_dtype(
            subgroup_cols
        ) or pd.api.types.is_categorical_dtype(subgroup_cols):
            nominal_values = unique_vals
            for val in nominal_values:
                desc_eq = seed_desc + [(col, ("=", val))]
                desc_neq = seed_desc + [(col, ("!=", val))]
                new_descriptions.append(desc_eq)
                new_descriptions.append(desc_neq)

            # print(f"Depth {level} Processing {col}. Values = {nominal_values}")

        # for numeric variables
        elif pd.api.types.is_numeric_dtype(subgroup_cols):
            sorted_vals = unique_vals
            n = len(sorted_vals)

            # in case only single value exists
            if n == 1:
                single_val = sorted_vals[0]
                new_descriptions.append(seed_desc + [(col, single_val)])
                # print(f"Depth {level} Processing {col}. Values = {single_val}")
            else:
                # for getting split points
                split_points = []
                for j in range(1, b):
                    idx = math.floor((j * n) / b) - 1
                    idx = max(0, min(idx, n - 1))
                    s_j = sorted_vals[idx]
                    split_points.append(s_j)

                for s_j in split_points:
                    desc_le = seed_desc + [(col, ("<=", s_j))]
                    desc_ge = seed_desc + [(col, (">=", s_j))]
                    new_descriptions.append(desc_le)
                    new_descriptions.append(desc_ge)

                # print(f"Depth {level} Processings {col}. Split points = {split_points}")

        # else:
        # print(f"Depth {level} Skipping {col}. - Error in data")

    # return new refiend descriptions
    return new_descriptions
