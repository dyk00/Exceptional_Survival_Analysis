# return sorted descriptions,
# to prevent duplicates when inserting a new description
def sort_description(description):
    return sorted(description, key=lambda x: (x[0], str(x[1])))


# to print a list of descriptions in a readable format
def description_to_string(description):

    # initalize string
    str = []

    # for each variable and value in the description
    for (var, value) in description:

        # if the value is a boolean
        if isinstance(value, bool):
            str.append(f"{var} == {value}")

        # if the value is a numeric range
        elif isinstance(value, tuple):
            l, r = value
            str.append(f"{var} in [{l}, {r})")

        # if the value is a single numeric value
        else:
            str.append(f"{var} == {value}")

    # return the combined string
    return " ^ ".join(str)
