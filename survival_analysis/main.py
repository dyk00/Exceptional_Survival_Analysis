# basic python
import numpy as np

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
    predict_cox_sksurv,
    test_proportional_hazards,
    check_cox_assumptions,
    predict_probability_cox,
    predict_hazard_cox,
)

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
from survival_analysis.evaluation.evaluation import get_c_index, get_c_index_ipcw


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

    # fit the cox on scikit-survival
    coxph_sksurv = fit_cox_sksurv(
        X_train, y_train_surv, alpha_min_ratio=0.05, l1_ratio=0.5
    )
    prediction_sksurv = predict_cox_sksurv(coxph_sksurv, X_test)

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
        min_time, max_time, min_event_time, max_event_time
    )

    # predict survival probabilities
    survival = predict_probability_cox(cph, X_test, time_grid)
    surv_probs = survival.T.to_numpy()

    # plot per individuals
    # or sample_size=len(survival)
    plot_survival_functions(survival, sample_size=5)

    # predict hazard scores
    # interchancable with 'prediction' using sksurv coxph
    hazard_scores = predict_hazard_cox(cph, X_test)

    # corcordance index for training set
    print("Concordance Index on Training Set:", cph.concordance_index_)

    # compute evaluation metrics
    c_index = get_c_index(test_df, duration_col, event_col, hazard_scores)
    print("Concordance Index (lifelines) on Test Set:", c_index)

    # # same thing
    # print("C-Index", cph.score(test_df, scoring_method="concordance_index"))

    # the usage:
    # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
    c_index_sk = concordance_index_censored(
        y_test_surv[event_col], y_test_surv[duration_col], prediction_sksurv
    )
    print(f"Concordance Index (sksurv) on Test Set: {c_index_sk[0]}")

    c_index_ipcw = get_c_index_ipcw(y_train_surv, y_test_surv, prediction_sksurv)
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

    # # get new test df after fitting cox to run emm afterwards
    # cox_with_prob = get_avg_hourly(test_df, survival, duration_col)
    # print(cox_with_prob.head())
    # # save_parquet(df, "./data", "cox_with_prob.parquet")


if __name__ == "__main__":
    main()
