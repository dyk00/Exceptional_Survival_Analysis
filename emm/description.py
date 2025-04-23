import numpy as np

# return sorted descriptions,
# to prevent duplicates when inserting a new description
# heap raise an error if priority is the same
def val_list(val):
    if isinstance(val, np.ndarray):
        return tuple(val.tolist())
    return val

# prevent duplications when having error just in case
def sort_description(description):
    return [(var, val_list(val)) for var, val in sorted(
        description, key=lambda x: (x[0], str(val_list(x[1])))
    )]

# to print a list of descriptions in a readable format
def description_to_string(description):

    # initalize string
    str = []

    # for each variable and value in the description
    for (var, value) in description:

        # if the value is a boolean
        if isinstance(value, bool):
            str.append(f"{var} == {value}")

        # if the value is a numeric or nominal as a single value
        elif isinstance(value, (int, float)):
            str.append(f"{var} == {value}")
            
        # if the value is a tuple specifying a range (<= or >=)
        # value[0] is symbol, value[1] is number
        elif isinstance(value, tuple):
            sym, num = value
            str.append(f"{var} {sym} {num}")

    # return the combined string
    return " ^ ".join(str)
