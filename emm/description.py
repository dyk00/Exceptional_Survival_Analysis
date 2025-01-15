# return sorted descriptions,
# to prevent duplicates when inserting a new description
def sort_description(description):
    return sorted(description, key=lambda x: (x[0], str(x[1])))
