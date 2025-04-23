import sys

sys.path.append("..")

# io
from data_processing.data_io import read_parquet, save_parquet

# emm
from emm.emm_beam import emm_beam
from emm.description import description_to_string
from emm.subgroup import filter_data

# aam
from aam.aam_modified import AAM


def main():

    df, _ = read_parquet(
        "..",
        "..",
    )
    df = df.drop(columns=["DateTime", "AdmissionID", "PatientID", "EID"])

    # just in case columns are still in boolean type,
    # but make sure there are no missing values
    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
    df[bool_cols] = df[bool_cols].astype(bool)

    # to exclude from features
    # we just call the aam_inverse as avg_surv_prob
    # as the emm is fit into the survival probabilty, we use inverse of aam (kind of risk score)
    # or -data['AAM']
    df["AAM_INVERSE"] = 100 - df["AAM"]
    df = df.drop(columns=["AAM"])

    df["avg_survival_probability"] = df["AAM_INVERSE"]
    df = df.drop(columns=["AAM_INVERSE"])
    quality_columns = ["avg_survival_probability"]

    # target variables
    target_columns = ["I_PRE", "I_PRE_V2"]

    # include all except for target and quality
    feature_columns = [
        c for c in df.columns if c not in target_columns and c not in quality_columns
    ]

    # based on which variables to create subgroups
    test_variables = [
        "SEX",
        "ADMIT1",
        "ADMIT2",
        "ADMIT3",
        "ADMIT4",
        "SEASON1",
        "SEASON2",
        "SEASON3",
        "DAYTIME1",
        "DAYTIME2",
        "DAYTIME3",
        "DAYTIME4",
        "CARE_ORDERDNR",
        "CARE_ORDERFULL_CODE",
        "CARE_ORDERPARTIAL",
        "I_OR_ACUTE",
        "I_OR_ELECTIVE",
        "I_OR_UNKNOWN",
        "I_IC",
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
                # constraints=None,
                b=5,
            )

            # if no subgroups are found
            if not result_set.heap:
                print("No subgroups that satisfy the constraints.")
                return

            # print the q numbers of subgroups that have the lowest survival probability on average
            print("\n===== Q-Subgroups with the Highest Risk Scores =====")
            for quality, _, description in sorted(result_set.heap, reverse=True):
                desc = description_to_string(description)
                subgroup = filter_data(description, df)
                print(
                    f"Description: {desc}, (Size = {len(subgroup)}), "
                    f"Quality (100-AMM): {-quality:.4f}, Quality (AMM): {100+quality:.4f}"
                )


if __name__ == "__main__":
    main()
