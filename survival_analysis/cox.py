# basic python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# emm
from emm.discretization import discretize_numeric_cols

# lifelines
from lifelines import KaplanMeierFitter
from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.utils import concordance_index, k_fold_cross_validation
from lifelines.statistics import proportional_hazard_test
from lifelines.calibration import survival_probability_calibration

# scikit-survival
from sksurv.metrics import (
    integrated_brier_score,
    cumulative_dynamic_auc,
    concordance_index_censored,
    concordance_index_ipcw,
)
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

# roc curve
from sklearn.metrics import roc_curve, auc

# data processing
from data_processing.data_io import read_parquet, save_parquet
from data_processing.train_processing import index_data, split_data


def get_time_range(test_df, duration_col, event_col):

    # time range for the test dataset
    min_time = test_df[duration_col].min()
    max_time = test_df[duration_col].max()

    # this is necessary to calculate the AUC and avoid dividing by 0
    event_times = test_df.loc[test_df[event_col] == True, duration_col]
    min_event_time = event_times.min() if len(event_times) > 0 else min_time
    max_event_time = event_times.max() if len(event_times) > 0 else max_time
    return min_time, max_time, min_event_time, max_event_time


# make df into survival form to use in scikit-survival
def make_surv(df, duration_col, event_col):
    return Surv.from_dataframe(event_col, duration_col, df)


# cox training
def fit_cox(train_df, duration_col, event_col, alpha=0.05, penalizer=0.01):
    cph = CoxPHFitter(alpha, penalizer)
    cph.fit(
        train_df,
        duration_col,
        event_col,
        batch_mode=True,
    )
    return cph


# predict survival probability S(t) (prob of no event occured (event-free) by t)
# failure probability S(t) = 1- F(t) (prob of event occured by t)
def predict_probability_cox(cph, X_test, times):
    surv_prob = cph.predict_survival_function(X_test, times=times)
    return surv_prob


# predict relative risk/hazard ratio for each individual
# so how much greater one’s hazard is relative to another
# but not probability
def predict_hazard_cox(cph, X_test):
    return cph.predict_partial_hazard(X_test)


# test proportional hazards
def test_proportional_hazards(cph, df):
    return proportional_hazard_test(cph, df, time_transform="rank")


# get concordance index on test set
# if the predicted scores are risks/hazards, multiply by -1
# https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
def get_c_index(test_df, duration_col, event_col, hazard_scores):
    return concordance_index(
        test_df[duration_col],
        -hazard_scores,
        test_df[event_col],
    )


# get concordance index based on inverse probability of censoring weights
def get_c_index_ipcw(y_train_surv, y_test_surv, hazard_scores):
    result = concordance_index_ipcw(y_train_surv, y_test_surv, hazard_scores)
    return result[0]


# plot coefficients with CI
def plot_coef_ci(cph):
    plt.figure(figsize=(6, 4))
    cph.plot()
    plt.show()


# check the assumptions and plot
def check_cox_assumptions(cph, train_df, thr=0.05, bool=False):
    cph.check_assumptions(train_df, p_value_threshold=thr, show_plots=bool)


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


# plot time dependent auc based on times with events
# to evaluate discriminative power
def plot_time_dependent_auc(event_time_grid, auc_scores, mean_auc_score):
    plt.figure(figsize=(6, 4))
    plt.plot(event_time_grid, auc_scores, marker="o", label="Time-dependent AUC")

    # add a line for mean auc
    plt.axhline(
        mean_auc_score,
        linestyle="--",
        color="red",
        label=f"Mean AUC = {mean_auc_score:.3f}",
    )
    plt.xlabel("Hours Since Admission")
    plt.ylabel("Time-dependent AUC")

    # plt.title("AUC over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


# plot time dependent roc curve
# roc curve does not break on a normal time grid
# use grid because survival is computed based on defined time grid
def plot_time_dependent_roc(
    eval_times, survival, time_grid, test_df, duration_col, event_col
):
    plt.figure(figsize=(6, 4))

    # for each evaluation time point,
    for t in eval_times:

        # compare the closest time in the time grid and the last index
        # and take the minimum as index
        idx = min(np.searchsorted(time_grid, t), len(time_grid) - 1)
        closest_time = time_grid[idx]

        # as event means deterioration event in our case,
        # get the risk score on the corresponding index

        event_probs = 1 - survival.iloc[idx, :].values

        # (time_to_event <= closest_time) and (event == True)
        label = (
            (test_df[duration_col].values <= closest_time)
            & (test_df[event_col].values == True)
        ).astype(bool)

        # compute the roc and auc
        fpr, tpr, _ = roc_curve(label, event_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"t={closest_time:.2f} (AUC={roc_auc:.3f})")

    # reference line
    plt.plot([0, 1], [0, 1], "k--", label="Random   (AUC=0.500)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

    # plt.title("Time-Dependent ROC Curves upto Multiple Time Points")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()


def plot_km(train_df, duration_col, event_col, strata=None):
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(6, 4))

    # plot different curves per strata
    if strata:
        groups = train_df[strata].unique()
        for group in groups:
            mask = train_df[strata] == group
            kmf.fit(
                train_df.loc[mask, duration_col],
                event_observed=train_df.loc[mask, event_col],
                label=str(group),
            )
            kmf.plot_survival_function(ci_show=True)
        # plt.title(f"Kaplan-Meier Survival Curves by {strata}")
    else:
        # single curve
        kmf.fit(
            train_df[duration_col],
            event_observed=train_df[event_col],
            label="Kaplan-Meier Estimate",
        )
        kmf.plot_survival_function(ci_show=True)
        # plt.title("Kaplan-Meier Survival Curve")

    plt.xlabel("Hours")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_survival_cols(cph, train_df, cols, bins_count):
    plt.figure(figsize=(8, 6))

    # get the discretized column bins
    col_bins = discretize_numeric_cols(train_df, cols, bins_count=bins_count)

    # if given string, then convert it as list
    if isinstance(cols, str):
        cols = [cols]

    # dict for storing values for each bin
    value = {}
    for col in cols:

        # if bool, then get F/T
        if train_df[col].dtype == "bool":
            value[col] = [False, True]
        else:
            # if numeric, get the values of the bins
            bin_ranges = col_bins.get(col)

            # we will only use lower bound value
            value[col] = [b[0] for b in bin_ranges]

    # get the combinations of values
    combs = list(product(*value.values()))
    values = [list(comb) for comb in combs]

    # get the partial effect of the combinations
    ax = cph.plot_partial_effects_on_outcome(cols, values, cmap="tab20")

    # get lines
    lines = ax.get_lines()

    # for each result and values,
    for line, comb in zip(lines, combs):
        labels = []

        # for each column and values,
        for col, val in zip(cols, comb):

            # if bool, then append T/F to labels
            if isinstance(val, bool):
                labels.append(f"{col}={'True' if val else 'False'}")
            else:
                # if numeric, append rounded value
                labels.append(f"{col}={val:.1f}")

        # merge labels
        label = ", ".join(labels)
        line.set_label(label)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Variables",
    )

    plt.title(f"Partial Effects of {', '.join(cols)} on Survival Outcome")
    plt.tight_layout()
    plt.show()


def main():
    # load and preprocess data
    df, _ = read_parquet("./data", "example_data_without_prob.parquet")
    df = df.drop(columns=["datetime"])

    # to remove the p_id and admission_id
    df = index_data(df)
    print(df)
    duration_col, event_col = "time_to_event", "is_first"

    # split the data into train and test sets
    train_df, test_df, X_train, X_test, y_train, y_test = split_data(
        df, duration_col, event_col
    )

    # make the survival form to use scikit-survival packages
    y_train_surv = make_surv(train_df, duration_col, event_col)
    y_test_surv = make_surv(test_df, duration_col, event_col)

    # fit the cox on scikit-survival
    coxph = CoxPHSurvivalAnalysis(alpha=0.05)
    coxph.fit(X_train, y_train_surv)
    prediction = coxph.predict(X_test)

    # fit the cox and test proportional hazards on lifelines
    cph = CoxPHFitter(penalizer=0.1, alpha=0.05)
    cph.fit(train_df, duration_col, event_col)
    print("Proportional Hazards Summary:")
    print(test_proportional_hazards(cph, train_df).summary)
    print(cph.print_summary())

    check_cox_assumptions(cph, train_df, bool=True)

    scores = k_fold_cross_validation(
        cph,
        train_df,
        duration_col,
        event_col,
        k=3,
        scoring_method="concordance_index",
        seed=0,
    )
    print("K-Fold Score: ", scores)
    print("K-Fold Mean Score", np.mean(scores))

    # plot coefficients with CI
    plot_coef_ci(cph)

    # get the time grid from the test set
    min_time, max_time, min_event_time, max_event_time = get_time_range(
        test_df, duration_col, event_col
    )

    survival_probability_calibration(cph, train_df, t0=100)

    # arbitrary time points
    n_timepoints = 50

    # ibs enforces all times must be < max_time
    # or could do np.linspace(min_time, max_time, num=n_timepoints, endpoint=False)
    time_grid = np.linspace(min_time, max_time - 1, num=n_timepoints, endpoint=True)

    # predict survival probabilities and risk scores
    survival = predict_probability_cox(cph, X_test, time_grid)
    cph.predict_survival_function(X_test[:5]).plot()

    # interchancable with 'prediction' using sksurv coxph
    hazard_scores = predict_hazard_cox(cph, X_test)
    surv_probs = survival.T.to_numpy()

    # corcordance index for training set
    print("Concordance Index on Training Set:", cph.concordance_index_)

    # compute evaluation metrics
    c_index = get_c_index(test_df, duration_col, event_col, hazard_scores)
    print("Concordance Index (lifelines) on Test Set:", c_index)

    # same thing
    # print("C-Index", cph.score(test_df, scoring_method="concordance_index"))

    # the usage:
    # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
    c_index_sk = concordance_index_censored(
        y_test_surv[event_col], y_test_surv[duration_col], prediction
    )
    print(f"Concordance Index (sksurv) on Test Set: {c_index_sk[0]}")

    c_index_ipcw = get_c_index_ipcw(y_train_surv, y_test_surv, prediction)
    print(f"IPCW Concordance Index: {c_index_ipcw:.4f}")

    # estimate should be the survival probabilites
    ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_probs, time_grid)
    print("Integrated Brier Score:", ibs)

    # define event time grid
    # so that when calculating auc scores, it won't get division by 0 error
    event_time_grid = np.linspace(
        min_event_time, max_event_time, num=n_timepoints, endpoint=True
    )
    # get time dependent auc
    auc_scores, mean_auc_score = cumulative_dynamic_auc(
        y_train_surv, y_test_surv, surv_probs, event_time_grid
    )
    print("Time-Dependent AUC scores:", auc_scores)
    print("Mean AUC score:", mean_auc_score)
    plot_time_dependent_auc(event_time_grid, auc_scores, mean_auc_score)

    # get time dependent roc curve for multiple time points
    # the time points will be adjusted to the defined time grid
    eval_times = [100, 300, 500, 700, 900]
    plot_time_dependent_roc(
        eval_times=eval_times,
        survival=survival,
        time_grid=time_grid,
        test_df=test_df,
        duration_col=duration_col,
        event_col=event_col,
    )

    # plot kaplan meier and can stratify by group
    plot_km(train_df, duration_col, event_col, strata="male")

    # plot survival curve for cox
    plot_survival_cols(cph, train_df, cols=["age", "male"], bins_count=5)

    # get new test df after fitting cox to run emm afterwards
    cox_with_prob = get_avg_hourly(test_df, survival, duration_col)
    print(cox_with_prob.head())
    # save_parquet(df, "./data", "cox_with_prob.parquet")


if __name__ == "__main__":
    main()
