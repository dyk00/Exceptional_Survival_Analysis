import pandas as pd

# check if the column/variable is already in the seed description
def bool_var_exists(description, col):
    for desc in description:
        # where desc is a tuple (var, value)
        if desc[0] == col:
            return True
    return False


# check if the column/variable and its range are already in the seed description
def num_var_range_exists(description, col, bin_range):
    for desc in description:
        if desc[0] == col and desc[1] == bin_range:
            return True
    return False


# add new descriptions to the seed description
# (e.g. [('age', [30, 41))] to [('age', [30, 41)), ('male', True)])
def refinement_operator(seed_desc, dataset, columns, col_bins):

    # initialize new_descriptions
    new_descriptions = []

    # for each column
    for col in columns:

        # if the column is boolean
        if pd.api.types.is_bool_dtype(dataset[col]):

            # if the column is not in the seed description yet
            if not bool_var_exists(seed_desc, col):

                # add the column with False and True to the existing seed description
                new_descriptions.append(seed_desc + [(col, False)])
                new_descriptions.append(seed_desc + [(col, True)])

        # if the column is numeric
        elif pd.api.types.is_numeric_dtype(dataset[col]):

            # if the column and its range are not in the seed description yet
            if not bool_var_exists(seed_desc, col):

                # add all possible ranges of this column from col_bins as a list
                possible_ranges = col_bins.get(col, [])

                # for each range in the possible ranges
                for r in possible_ranges:

                    # if the column and its range is not in the seed description
                    if not num_var_range_exists(seed_desc, col, r):

                        # add the range to the existing seed description
                        new_descriptions.append(seed_desc + [(col, r)])

    # return new refiend descriptions
    return new_descriptions
