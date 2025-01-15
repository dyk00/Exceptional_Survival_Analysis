# io
from data_processing.data_io import read_parquet

# emm
from emm.discretization import discretize_numeric_cols
from emm.emm_beam import emm_beam
from emm.description import description_to_string

# survival analysis


def main():
    df, cols = read_parquet("./data", "example_data_with_prob.parquet")

    # just in case columns are still in boolean type,
    # but make sure there are no missing values
    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
    df[bool_cols] = df[bool_cols].astype(bool)

    # target variables are same as survival analysis ones
    target_columns = ["time_to_event", "is_first"]

    # to exclude from features
    quality_columns = ["avg_survival_probability"]

    # include all except for target and quality
    feature_columns = [
        c for c in cols if c not in target_columns and c not in quality_columns
    ]

    # discretize numerical columns
    col_bins = discretize_numeric_cols(df, feature_columns, bins_count=5)

    # define constraints (e.g. min_size, max_size)
    constraints = {"min_size": 3}

    # run emm_beam to get the result set
    result_set = emm_beam(
        dataset=df,
        columns=feature_columns,
        col_bins=col_bins,
        # beam width
        w=8,
        # maximum depth to explore
        d=3,
        # how many subgroups in result set
        q=5,
        constraints=constraints,
    )

    # if no subgroups are found
    if not result_set.heap:
        print("No subgroups that satisfy the constraints.")
        return

    # print the q numbers of subgroups that have the lowest survival probability on average
    print("\n===== Q-Subgroups with the Lowest Survival Probability =====")
    for quality, description in sorted(result_set.heap, reverse=True):
        desc = description_to_string(description)
        print(
            f"Description: {desc}, Quality (Average Survival Probability): {-quality:.4f}"
        )


if __name__ == "__main__":
    main()
