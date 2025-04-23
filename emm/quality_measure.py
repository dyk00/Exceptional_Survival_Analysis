from .subgroup import filter_data

# given the new description,
# calculate the average survival probability of the subgroup
def quality_measure(description, dataset):
    subgroup = filter_data(description, dataset)

    # assign highest value (survival probability) to empty subgroups
    # so that non-empty subgroups will have higher priority
    # as it works as negative value within the algorithm
    if len(subgroup) == 0:
        return float("inf")
    return subgroup["avg_survival_probability"].mean()
