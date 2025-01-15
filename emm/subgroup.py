import pandas as pd

# filter data that satisfies the description to return subgroups
def filter_data(description, data):

    # if non specified, then just return the whole data
    if not description:
        return data

    # initialize boolean mask for filtering data
    # to indicate whether each row meets the all descriptions
    mask = pd.Series(True, index=data.index)

    # description is a list of tuples (var, value)
    # (e.g. [('age', [30, 41)), ('male', True)])
    # value can be a single (bool/numeric) value True, (41) or range [30, 41)
    # e.g. var = 'age', value = [30, 41)
    for var, value in description:

        # check if the value is the same as the description's value (e.g. True)
        if isinstance(value, bool):
            mask = mask & (data[var] == value)

        # check whether the actual value is
        # in the range of the description's value (e.g. [30, 41))
        elif isinstance(value, tuple):
            l, r = value
            mask = mask & (data[var] >= l) & (data[var] < r)

        # check if the value is the same as the description's value (e.g. 41)
        else:
            mask = mask & data[var] == value

    # return the filtered data (subgroup)
    return data[mask]
