from .subgroup import filter_data

# check whether a given description satisfies all specified constraints
# constraints are in the form of dictionary
def satisfies_all(description, data, constraints=None):

    # if non specified, then automatically True
    if not constraints:
        return True

    # filter data that satisfies the description so we don't need to check all data
    subgroup = filter_data(description, data)

    # for each constraint
    for col, value in constraints.items():
        if col in ["min_size", "max_size"]:
            continue

        # if the constraint is in tuple
        if isinstance(value, tuple) and len(value) == 2:
            l, r = value
            # check if the value is in the range
            if not ((subgroup[col] >= l) & (subgroup[col] <= r)).all():
                return False
        else:
            # or check exact value
            if not (subgroup[col] == value).all():
                return False

    # minimum size (for now)
    if "min_size" in constraints:
        min_size = constraints["min_size"]
        if len(subgroup) < min_size:
            return False

    # maximum size
    if "max_size" in constraints:
        max_size = constraints["max_size"]
        if len(subgroup) > max_size:
            return False

    # if all constraints are satisfied, then return True
    return True
