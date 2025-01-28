# basic python
import numpy as np
import pandas as pd

# lifelines
from lifelines.utils import k_fold_cross_validation
from lifelines.calibration import survival_probability_calibration

# scikit-survival
from sksurv.metrics import (
    integrated_brier_score,
    cumulative_dynamic_auc,
    concordance_index_censored,
)

# data processing
from data_processing.data_io import read_parquet, save_parquet
from data_processing.train_processing import index_data, split_data

# survival utilities
from data_processing.surv_util import (
    get_time_range,
    get_time_grids,
    make_surv,
    get_avg_hourly,
)

# cox
from survival_analysis.models.cox import (
    fit_cox_lifelines,
    fit_cox_sksurv,
    predict_hazard_cox_sksurv,
    predict_probability_cox_sksurv,
    predict_probability_cox_lifelines,
    predict_hazard_cox_lifelines,
    test_proportional_hazards,
    check_cox_assumptions,
)

# logistic regression
from survival_analysis.models.lr import (
    fit_lr,
    predict_lr,
    evaluate_lr,
    plot_confusion_matrix,
    plot_roc_curve,
)

# gradient boosting
from survival_analysis.models.gb import fit_gb, fit_cgb, predict_hazard_sksurv

# plot
from survival_analysis.evaluation.plot import (
    plot_coef_ci,
    plot_survival_functions,
    plot_time_dependent_auc,
    plot_time_dependent_roc,
    plot_km,
    plot_survival_cols,
)

# evaluation
from survival_analysis.evaluation.evaluation import (
    get_c_index_lifelines,
    get_c_index_sksurv,
    get_c_index_ipcw,
)


def main():
    # load and preprocess data
    df, _ = read_parquet("./data", "example_data_without_prob.parquet")
    df = df.drop(columns=["datetime"])

    # to remove the p_id and admission_id
    df = index_data(df)
    duration_col, event_col = "time_to_event", "is_first"

    # split the data into train and test sets
    train_df, test_df, X_train, X_test, y_train, y_test = split_data(
        df, duration_col, event_col
    )

    # make the survival form to use scikit-survival packages
    y_train_surv = make_surv(train_df, duration_col, event_col)
    y_test_surv = make_surv(test_df, duration_col, event_col)

    # ------------------- Cox proportional-hazards ------------------- #

    # fit the cox on scikit-survival
    coxph_sksurv = fit_cox_sksurv(
        X_train, y_train_surv, alpha_min_ratio=0.05, l1_ratio=0.5
    )

    # get hazard/risk scores
    hazard_sksurv = predict_hazard_cox_sksurv(coxph_sksurv, X_test)

    # get survival probabilities
    surv_probs_sksurv = predict_probability_cox_sksurv(coxph_sksurv, X_test)

    # fit the cox and test proportional hazards on lifelines
    cph = fit_cox_lifelines(train_df, duration_col, event_col)
    print("Proportional Hazards Summary:")
    print(test_proportional_hazards(cph, train_df).summary)
    print(cph.print_summary())

    # check the assumptions of cox
    # change the bool to True to see the plots
    check_cox_assumptions(cph, train_df, bool=True)

    # k fold cross validation using c-index
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

    # smoothed calibration curves, evaulating at time point 100
    survival_probability_calibration(cph, train_df, t0=100)

    # get the time grids based on time ranges
    time_grid, event_time_grid = get_time_grids(
        min_time, max_time, min_event_time, max_event_time, n_timepoints=50
    )

    # predict survival probabilities
    survival = predict_probability_cox_lifelines(cph, X_test, time_grid)
    surv_probs = survival.T.to_numpy()

    # plot per individuals
    # or sample_size=len(survival)
    plot_survival_functions(survival, sample_size=5)

    # predict hazard scores
    # interchancable with 'prediction' using sksurv coxph
    hazard_scores = predict_hazard_cox_lifelines(cph, X_test)

    # corcordance index for training set
    print("Concordance Index on Training Set:", cph.concordance_index_)

    # compute evaluation metrics
    c_index = get_c_index_lifelines(test_df, duration_col, event_col, hazard_scores)
    print("Concordance Index (lifelines) on Test Set:", c_index)

    # # same thing
    # print("C-Index", cph.score(test_df, scoring_method="concordance_index"))

    # the usage:
    # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
    c_index_sk = concordance_index_censored(
        y_test_surv[event_col], y_test_surv[duration_col], hazard_sksurv
    )
    print(f"Concordance Index (sksurv) on Test Set: {c_index_sk[0]}")

    c_index_ipcw = get_c_index_ipcw(y_train_surv, y_test_surv, hazard_sksurv)
    print(f"IPCW Concordance Index: {c_index_ipcw:.4f}")

    # estimate should be the survival probabilites
    ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_probs, time_grid)
    print("Integrated Brier Score:", ibs)

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
    cox_with_prob = test_df.copy()
    cox_with_prob = get_avg_hourly(cox_with_prob, survival, duration_col)
    # save_parquet(df, "./data", "cox_with_prob.parquet")

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

    # ------------------- Gradient Boosting ------------------- #

    # gradient boosting
    gb = fit_gb(X_train, y_train_surv)

    # # componentwise gradient boosting
    # cgb = fit_cgb(X_train, y_train_surv)

    # fit gradient boosting.
    # can replace with componentwise gradient boosting
    gb = fit_gb(X_train, y_train_surv)

    # predict hazard (risk) scores
    hazard_sksurv = predict_hazard_sksurv(gb, X_test)

    # get a list of StepFunctions (n_samples,)
    step_funcs = gb.predict_survival_function(X_test, return_array=False)

    # map each StepFunction on time_grid
    # shape (n_samples, n_timepoints)
    surv_probs = np.array([sf(time_grid) for sf in step_funcs])

    # shape = (n_timepoints,  n_samples)
    survival = pd.DataFrame(data=surv_probs.T, index=time_grid)
    plot_survival_functions(survival, sample_size=5)

    # get c-index
    gb_c_index = get_c_index_sksurv(gb, X_test, y_test_surv)
    print("GB C-index:", gb_c_index)

    # get ipcw c-index
    c_index_ipcw = get_c_index_ipcw(y_train_surv, y_test_surv, hazard_sksurv)
    print("GB IPCW c-index:", c_index_ipcw)

    # get IBS
    ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_probs, time_grid)
    print("Integrated Brier Score:", ibs)

    # plot time-dependent AUC
    auc_scores, mean_auc_score = cumulative_dynamic_auc(
        y_train_surv, y_test_surv, surv_probs, event_time_grid
    )
    print("Time-Dependent AUC scores:", auc_scores)
    print("Mean AUC score:", mean_auc_score)
    plot_time_dependent_auc(event_time_grid, auc_scores, mean_auc_score)

    # plot time dependent roc curve
    eval_times = [100, 300, 500, 700, 900]
    plot_time_dependent_roc(
        eval_times=eval_times,
        survival=survival,
        time_grid=time_grid,
        test_df=test_df,
        duration_col="time_to_event",
        event_col="is_first",
    )

    # ------------------- Random Survival Forest ------------------- #


if __name__ == "__main__":
    main()
