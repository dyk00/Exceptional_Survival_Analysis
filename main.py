import sys

sys.path.append("..")

# io
from data_processing.data_io import read_parquet

# emm
from emm.emm_beam import emm_beam
from emm.description import description_to_string
from emm.subgroup import filter_data


def main():
    df, cols = read_parquet(
        "..",
        "weibull_with_prob.parquet",
    )

    df = df.drop(columns=["p_id", "opname_id", "hourly_probabilities"])
    # print(df['leeftijd'].value_counts())
    # print(df.head())

    # just in case columns are still in boolean type,
    # but make sure there are no missing values
    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
    df[bool_cols] = df[bool_cols].astype(bool)
    df[["geslacht", "spoed"]] = df[["geslacht", "spoed"]].astype(bool)

    # update cols
    cols = list(df.columns)

    # target variables are same as survival analysis ones
    target_columns = ["time_to_first_event", "is_first"]

    # to exclude from features
    quality_columns = ["avg_survival_probability"]

    # include all except for target and quality
    feature_columns = [
        c for c in cols if c not in target_columns and c not in quality_columns
    ]

    # based on which variables to create subgroups
    test_variables = [
        "geslacht",
        "spoed",
        "specialisme_code",
        "hoofdverrichting_code",
        "first_event",
        "care_order",
        "prioriteit_code",
    ]

    # define constraints (e.g. min_size, max_size)
    for var in test_variables:
        all_values = df[var].unique()

        for v in all_values:
            print(f"\n{var} = {v}")
            constraints = {var: v, "min_size": 274}

            # run emm_beam to get the result set
            result_set = emm_beam(
                dataset=df,
                columns=feature_columns,
                # beam width
                w=100,
                # maximum depth to explore
                d=2,
                # how many subgroups in result set
                q=10,
                constraints=constraints,
                # constraints=None
                # number of bins to discretize
                b=5,
            )

            # if no subgroups are found
            if not result_set.heap:
                print("No subgroups that satisfy the constraints.")
                continue

            # print the q numbers of subgroups that have the lowest survival probability on average
            print("\n===== Q-Subgroups with the Lowest Survival Probability =====")
            for quality, _, description in sorted(result_set.heap, reverse=True):
                desc = description_to_string(description)
                subgroup = filter_data(description, df)
                ##print(subgroup.head())
                print(
                    f"Description: {desc}, (Size = {len(subgroup)}), Quality (Average Survival Probability): {-quality:.4f}"
                )


if __name__ == "__main__":
    main()
