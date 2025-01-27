# basic python
import numpy as np
import pandas as pd

# scikit-survival
from sksurv.util import Surv


def get_time_range(test_df, duration_col, event_col):

    # time range for the test dataset
    min_time = test_df[duration_col].min()
    max_time = test_df[duration_col].max()

    # this is necessary to calculate the AUC and avoid dividing by 0
    event_times = test_df.loc[test_df[event_col] == True, duration_col]
    min_event_time = event_times.min() if len(event_times) > 0 else min_time
    max_event_time = event_times.max() if len(event_times) > 0 else max_time
    return min_time, max_time, min_event_time, max_event_time


# get times based on arbitrary time points
def get_time_grids(min_time, max_time, min_event_time, max_event_time, n_timepoints=50):

    # ibs enforces all times must be < max_time
    # or could do np.linspace(min_time, max_time, num=n_timepoints, endpoint=False)
    time_grid = np.linspace(min_time, max_time - 1, num=n_timepoints, endpoint=True)

    # define event time grid
    # so that when calculating auc scores, it won't get division by 0 error
    event_time_grid = np.linspace(
        min_event_time, max_event_time, num=n_timepoints, endpoint=True
    )
    return time_grid, event_time_grid


# make df into survival form to use in scikit-survival
def make_surv(df, duration_col, event_col):
    return Surv.from_dataframe(event_col, duration_col, df)


def get_avg_survival(row, survival, duration_col):
    patient = row.name
    event_time = row[duration_col]

    # if index exists in survival col
    if patient in survival.columns and not pd.isna(event_time):

        # get survival probabilities up to the time to event
        probs = survival.loc[survival.index <= event_time, patient]

        # calculate the average probability per patient
        if not probs.empty:
            return probs.mean()
    return np.nan


def get_hourly_probability(row, survival, duration_col):
    patient = row.name
    event_time = row[duration_col]

    if patient in survival.columns and not pd.isna(event_time):
        probs = survival.loc[survival.index <= event_time, patient]
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
