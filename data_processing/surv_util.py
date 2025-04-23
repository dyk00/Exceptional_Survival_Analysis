# basic python
import numpy as np
import pandas as pd

# scikit-survival
from sksurv.util import Surv

from sklearn.model_selection import train_test_split


def split_data(df, duration_col, event_col, test_size=0.2, random_state=42):
    train_val, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[event_col]
    )

    train_df, val_df = train_test_split(
        train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=train_val[event_col],
    )

    X_train = train_df.drop(
        columns=["p_id", "opname_id", event_col, duration_col], errors="ignore"
    )
    y_train = train_df[[event_col, duration_col]]

    X_val = val_df.drop(
        columns=["p_id", "opname_id", event_col, duration_col], errors="ignore"
    )
    y_val = val_df[[event_col, duration_col]]

    X_test = test_df.drop(
        columns=["p_id", "opname_id", event_col, duration_col], errors="ignore"
    )
    y_test = test_df[[event_col, duration_col]]

    return train_df, val_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test


def get_time_range(df, duration_col, event_col):

    # time range for the dataset
    min_time = df[duration_col].min()
    max_time = df[duration_col].max()

    # this is necessary to calculate the AUC and avoid dividing by 0
    event_times = df.loc[df[event_col] == True, duration_col]
    min_event_time = event_times.min() if len(event_times) > 0 else min_time
    max_event_time = event_times.max() if len(event_times) > 0 else max_time
    return min_time, max_time, min_event_time, max_event_time


def put_time_to_grid(df, duration_col, event_col, end_time=24, step=1):
    min_time, max_time, min_event_time, max_event_time = get_time_range(
        df, duration_col, event_col
    )
    return get_time_grids(
        min_time, max_time, min_event_time, max_event_time, end_time, step
    )


def get_time_grids(min_time, max_time, min_event_time, max_event_time, end_time, step):
    time_grid = np.arange(min_time, end_time + step, step)
    event_time_grid = np.arange(min_event_time, end_time + step, step)
    return time_grid, event_time_grid


# make df into survival form to use in scikit-survival
def make_surv(df, duration_col, event_col):
    return Surv.from_dataframe(event_col, duration_col, df)


def get_avg_survival(row, survival, duration_col):
    admission = row.name
    event_time = row[duration_col]

    # if index exists in survival col
    if admission in survival.columns and not pd.isna(event_time):

        # get survival probabilities up to the time to event
        probs = survival.loc[survival.index <= event_time, admission]

        # calculate the average probability per admission
        if not probs.empty:
            return probs.mean()
    return np.nan


def get_hourly_probability(row, survival, duration_col):
    admission = row.name
    event_time = row[duration_col]

    if admission in survival.columns and not pd.isna(event_time):
        probs = survival.loc[survival.index <= event_time, admission]
        if not probs.empty:

            # get each probability
            return probs.values.tolist()
    return None


# get the average and hourly probability
def get_avg_hourly(df, survival, duration_col):
    survival = survival.sort_index()

    df["avg_survival_probability"] = df.apply(
        lambda row: get_avg_survival(row, survival, duration_col), axis=1
    )

    df["hourly_probabilities"] = df.apply(
        lambda row: get_hourly_probability(row, survival, duration_col), axis=1
    )

    return df
