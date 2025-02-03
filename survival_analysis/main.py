# data processing
from data_processing.data_io import read_parquet

# survival utilities
from data_processing.surv_util import (
    index_data,
    split_data,
    get_time_range,
    get_time_grids,
    make_surv,
)

# kaplan-meier
from survival_analysis.plot import plot_km

# logistic regression
from survival_analysis.lr import *

# other models
from survival_analysis.fit import *
from survival_analysis.evaluate_lifelines import *
from survival_analysis.evaluate_sksurv import *

# kfold validation
from survival_analysis.evaluate_lifelines_kfold import *
from survival_analysis.evaluate_sksurv_kfold import *


def main():
    # load and preprocess data
    df, _ = read_parquet("./data", "example_data_without_prob.parquet")
    df = df.drop(columns=["datetime"])

    # to remove the p_id and admission_id
    df = index_data(df)
    duration_col, event_col = "time_to_event", "is_first"

    # ------------------- Define and Get variables ------------------- #

    # load and preprocess data
    df, _ = read_parquet("./data", "example_data_without_prob.parquet")
    df = df.drop(columns=["datetime"])

    # to remove the p_id and admission_id
    df = index_data(df)
    duration_col, event_col = "time_to_event", "is_first"

    # get features and target for k fold
    X = df.drop(columns=[duration_col, event_col]).copy()
    y = df[event_col].astype(int)

    # make the survival form to use scikit-survival packages
    y_surv = make_surv(df, duration_col, event_col)

    # split the data into train and test sets
    train_df, test_df, X_train, X_test, y_train, y_test = split_data(
        df, duration_col, event_col
    )

    # make the survival form to use scikit-survival packages
    y_train_surv = make_surv(train_df, duration_col, event_col)
    y_test_surv = make_surv(test_df, duration_col, event_col)

    # get the time grid from the test set
    min_time, max_time, min_event_time, max_event_time = get_time_range(
        test_df, duration_col, event_col
    )

    # get the time grids based on time ranges
    time_grid, event_time_grid = get_time_grids(
        min_time, max_time, min_event_time, max_event_time, n_timepoints=50
    )

    # ------------------- Kaplan Meier ------------------- #

    # plot kaplan meier and can stratify by group
    plot_km(train_df, duration_col, event_col, strata="male")

    # ------------------- Logistic Regression ------------------- #

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
        model_names=["cox_lf", "weibull", "ln", "ll"],
        train_df=train_df,
        test_df=test_df,
        X_test=X_test,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        duration_col=duration_col,
        event_col=event_col,
        time_grid=time_grid,
        event_time_grid=event_time_grid,
    )

    # ------------------- Scikit-Survival ------------------- #

    evaluate_sksurv(
        model_names=["cox_sk", "coxnet_sk", "gb", "gcb", "rsf", "ersf"],
        test_df=test_df,
        X_train=X_train,
        X_test=X_test,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        event_col=event_col,
        duration_col=duration_col,
        time_grid=time_grid,
        event_time_grid=event_time_grid,
    )

    # ------------------- K-Fold Validation for Lifelines ------------------- #
    # model_names = ["cox_lf", "weibull", "ln", "ll"]
    # kfold_results = evaluate_lifelines_kfold(
    #     model_names=model_names,
    #     df=df,
    #     X=X,
    #     y=y_surv,
    #     duration_col=duration_col,
    #     event_col=event_col,
    #     n_splits=5,
    #     random_state=42,
    # )

    # print("KFold Results:")
    # for model, metrics in kfold_results.items():
    #     print(f"Model: {model}")
    #     for metric_name, metric_val in metrics.items():
    #         print(f"{metric_name}: {metric_val}")

    # ------------------- K-Fold Validation for Scikit-Survival ------------------- #
    model_names = ["cox_sk", "coxnet_sk", "gb", "gcb", "rsf", "ersf"]
    kfold_results = evaluate_sksurv_kfold(
        model_names=model_names,
        df=df,
        X=X,
        y=y_surv,
        duration_col=duration_col,
        event_col=event_col,
        n_splits=5,
        random_state=42,
    )

    print("KFold Results:")
    for model, metrics in kfold_results.items():
        print(f"Model: {model}")
        for metric_name, metric_val in metrics.items():
            print(f"{metric_name}: {metric_val}")


if __name__ == "__main__":
    main()
