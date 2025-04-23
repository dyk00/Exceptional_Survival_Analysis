import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import seaborn as sns

# data processing
from data_processing.data_io import read_parquet

# survival utilities
from data_processing.surv_util import (
    split_data,
    get_time_range,
    get_time_grids,
    put_time_to_grid,
    make_surv,
)

# kaplan-meier and distribution plots
from survival_analysis.plot import (
    plot_km,
    plot_distribution_is_first_under24,
    plot_distribution_is_first_entire,
    plot_distribution_first_event_under24,
    plot_value_counts_histogram,
)

# logistic regression
from survival_analysis.lr import *

# other models
from survival_analysis.fit import *
from survival_analysis.evaluate_lifelines import *
from survival_analysis.evaluate_sksurv import *
from survival_analysis.plot_lifelines import *
from survival_analysis.plot_sksurv import *

# including index for training in lifelines
import warnings

warnings.simplefilter("ignore")


def main():

    # ------------------- Define and Get variables ------------------- #

    # load and preprocess data
    df, _ = read_parquet(
        "..",
        "df_for_sa_rand1.parquet",
    )

    # to set index
    df = df.set_index(["p_id", "opname_id"])
    duration_col, event_col = "time_to_first_event", "is_first"

    # print how many events/censored
    n_events = df[event_col].sum()
    n_censored = len(df) - n_events
    print("Num of events:", n_events)
    print("Num of censored:", n_censored)

    # filter all events under 24
    under_t = df[df[duration_col] < 24]
    print(f"Under t:", under_t.shape[0])

    total = df.shape[0]
    percent_t = (under_t.shape[0] / total) * 100
    print(f"Percent:", percent_t)

    # plot event distribution over time
    plot_distribution_is_first_under24(under_t, duration_col, event_col)
    plot_distribution_is_first_entire(df, duration_col, event_col)
    plot_distribution_first_event_under24(
        under_t, duration_col, first_event_col="first_event"
    )

    # plot the duration distribution
    plot_value_counts_histogram(df, duration_col, output_dir=DIR)

    # duration 0 happens because of the '12 hours before thing'.
    # this is for prevention in AFT models
    epsilon = 1e-6
    df[duration_col] = df[duration_col].apply(lambda t: epsilon if t == 0 else t)

    # make the survival form to use scikit-survival packages
    y_surv = make_surv(df, duration_col, event_col)

    # split the data into train and test sets
    (
        train_df,
        val_df,
        test_df,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df, duration_col, event_col)

    # make the survival form to use scikit-survival packages
    y_train_surv = make_surv(train_df, duration_col, event_col)
    y_test_surv = make_surv(test_df, duration_col, event_col)
    y_val_surv = make_surv(val_df, duration_col, event_col)

    time_grid_train, event_time_grid_train = put_time_to_grid(
        train_df, duration_col, event_col, end_time=24, step=1
    )
    time_grid_val, event_time_grid_val = put_time_to_grid(
        test_df, duration_col, event_col, end_time=24, step=1
    )
    time_grid_test, event_time_grid_test = put_time_to_grid(
        val_df, duration_col, event_col, end_time=24, step=1
    )

    # ------------------- Kaplan Meier ------------------- #

    # plot kaplan meier and can stratify by group
    plot_km(train_df, duration_col, event_col, strata="geslacht")
    plot_km(train_df, duration_col, event_col, strata="spoed")
    plot_km(train_df, duration_col, event_col, strata="specialisme_code")
    plot_km(train_df, duration_col, event_col, strata="hoofdverrichting_code")
    plot_km(train_df, duration_col, event_col, strata="count_death_fullcode")
    plot_km(train_df, duration_col, event_col, strata="count_death_ic")
    plot_km(train_df, duration_col, event_col, strata="count_ic_6hr")
    plot_km(train_df, duration_col, event_col, strata="count_acute_ic")
    plot_km(train_df, duration_col, event_col, strata="first_event")
    plot_km(train_df, duration_col, event_col, strata="m_year")
    plot_km(train_df, duration_col, event_col, strata="m_month")
    plot_km(train_df, duration_col, event_col, strata="m_day")
    plot_km(train_df, duration_col, event_col, strata="m_hour")

    #     # ------------------- Logistic Regression ------------------- #

    # fit standard logistic regression
    lr = fit_lr(X_train, y_train, duration_col, event_col)

    # predict and get a single probability per row
    lr_with_prob = test_df.copy()
    _, _, lr_with_prob = predict_lr(lr, X_test, lr_with_prob)

    # evluate metrics and get typecasted variables
    y_true, y_pred, y_pred_prob = evaluate_lr(lr, X_test, y_test, test_df, event_col)

    # plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # plot ROC curve
    plot_roc_curve(y_true, y_pred_prob)

    # ------------------- Lifelines ------------------- #

    evaluate_lifelines(
        model_names=["cox_lf", "weibull"],
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        y_val_surv=y_val_surv,
        duration_col=duration_col,
        event_col=event_col,
        time_grid_train=time_grid_train,
        event_time_grid_train=event_time_grid_train,
        time_grid_test=time_grid_test,
        event_time_grid_test=event_time_grid_test,
        time_grid_val=time_grid_val,
        event_time_grid_val=event_time_grid_val,
    )

    # ------------------- Scikit-Survival ------------------- #

    evaluate_sksurv(
        # "gb", "gcb" won't run in large data, cox_sk being strange
        # model_names=["cox_sk", "coxnet_sk", "gb", "gcb", "rsf", "ersf"],
        model_names=["coxnet_sk", "rsf", "ersf"],
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        y_val_surv=y_val_surv,
        duration_col=duration_col,
        event_col=event_col,
        time_grid_train=time_grid_train,
        event_time_grid_train=event_time_grid_train,
        time_grid_test=time_grid_test,
        event_time_grid_test=event_time_grid_test,
        time_grid_val=time_grid_val,
        event_time_grid_val=event_time_grid_val,
    )

    # ------------------- Lifelines Partial Effects ------------------- #

    df, _ = read_parquet("..", "df_for_sa.parquet")

    # to set index
    df = df.set_index(["p_id", "opname_id"])
    df = df.reset_index(drop=True)

    duration_col, event_col = "time_to_first_event", "is_first"

    # duration 0 happens because of the '12 hours before thing'.
    # this is for prevention in AFT models
    epsilon = 1e-6
    df[duration_col] = df[duration_col].apply(lambda t: epsilon if t == 0 else t)

    # make the survival form to use scikit-survival packages
    y_surv = make_surv(df, duration_col, event_col)

    # split the data into train and test sets
    (
        train_df,
        val_df,
        test_df,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df, duration_col, event_col)

    # make the survival form to use scikit-survival packages
    y_train_surv = make_surv(train_df, duration_col, event_col)
    y_test_surv = make_surv(test_df, duration_col, event_col)
    y_val_surv = make_surv(val_df, duration_col, event_col)

    plot_lifelines(
        model_names=["cox_lf", "weibull"],
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        y_val_surv=y_val_surv,
        duration_col=duration_col,
        event_col=event_col,
        time_grid_train=time_grid_train,
        event_time_grid_train=event_time_grid_train,
        time_grid_test=time_grid_test,
        event_time_grid_test=event_time_grid_test,
        time_grid_val=time_grid_val,
        event_time_grid_val=event_time_grid_val,
    )

    # ------------------- Scikit-Survival Partial Effects ------------------- #

    plot_sksurv(
        # "gb", "gcb" won't run in large data, cox_sk being strange
        # model_names=["cox_sk", "coxnet_sk", "gb", "gcb", "rsf", "ersf"],
        model_names=["coxnet_sk", "rsf", "ersf"],
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        y_val_surv=y_val_surv,
        duration_col=duration_col,
        event_col=event_col,
        time_grid_train=time_grid_train,
        event_time_grid_train=event_time_grid_train,
        time_grid_test=time_grid_test,
        event_time_grid_test=event_time_grid_test,
        time_grid_val=time_grid_val,
        event_time_grid_val=event_time_grid_val,
    )


if __name__ == "__main__":
    main()
