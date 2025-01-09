# io
from data_processing.data_io import read_parquet

# emm
from emm.discretization import discretize_numeric_cols
from emm.emm_beam import emm_beam
from emm.quality_measure import quality_measure
from emm.refinement import refinement_operator

# survival analysis


def main():

    df, cols = read_parquet("./data", "example_data.parquet")

    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns

    df[bool_cols] = df[bool_cols].astype(bool)

    target_columns = ["time_to_first_event", "avg_survival_probability"]

    feature_columns = [c for c in cols if c not in target_columns]

    col_bins = discretize_numeric_cols(df, feature_columns, bins_count=2)

    constraints = 3

    result_set = emm_beam(
        dataset=df,
        quality_measure=quality_measure,
        refinement_operator=refinement_operator,
        w=2,
        d=2,
        q=2,
        constraints=constraints,
    )


if __name__ == "__main__":
    main()
