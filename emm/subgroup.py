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
    for var, value in description:

        # check if the value is the same as the description's value (e.g. True)
        if isinstance(value, bool):
            mask = mask & (data[var] == value)

        # check whether the actual value is same as
        # the nomial or single value numeric value
        elif isinstance(value, (int, float, str)):
            mask = mask & (data[var] == value)

        # e.g. value = ("<=", s_j) or (">=", s_j)
        elif isinstance(value, tuple) and len(value) == 2:
            sym, num = value
            if sym == "<=":
                mask = mask & (data[var] <= num)
            elif sym == ">=":
                mask = mask & (data[var] >= num)
            else:
                # if ("=", something) or ("!=", something)
                if sym == "!=":
                    mask = mask & (data[var] != num)
                elif sym == "=":
                    mask = mask & (data[var] == num)

    # return the filtered data (subgroup)
    return data[mask]
