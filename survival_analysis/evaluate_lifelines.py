import numpy as np

from data_processing.data_io import save_parquet
from data_processing.surv_util import get_avg_hourly
from survival_analysis.fit import fit_cox_lf, fit_weibull, fit_ln, fit_ll
from survival_analysis.plot import *

from lifelines.utils import k_fold_cross_validation, concordance_index
from lifelines.calibration import survival_probability_calibration
from lifelines.statistics import proportional_hazard_test

from sksurv.metrics import (
    integrated_brier_score,
    cumulative_dynamic_auc,
    concordance_index_censored,
    concordance_index_ipcw,
)

FIT_FUNCTIONS1 = {
    "cox_lf": fit_cox_lf,
    "weibull": fit_weibull,
    "ln": fit_ln,
    "ll": fit_ll,
}


def evaluate_lifelines(
    model_names,
    train_df,
    test_df,
    X_test,
    y_train_surv,
    y_test_surv,
    duration_col,
    event_col,
    time_grid,
    event_time_grid,
):
    # global FIT_FUNCTIONS1

    for model_name in model_names:
        print(f"Evaluating : {model_name}")

        # lookup model_name
        fit_function = FIT_FUNCTIONS1[model_name]

        # fit the model
        if model_name == "cox_lf":
            model = fit_function(
                train_df, duration_col, event_col, alpha=0.05, penalizer=0.01
            )
        else:
            model = fit_function(train_df, duration_col, event_col)

        # print the summary
        print(model.print_summary())

        # print the proportional hazards summary and check assumptions
        if model_name == "cox_lf":
            print("Proportional Hazards Summary:\n")
            result = proportional_hazard_test(model, train_df, time_transform="rank")
            print(result.print_summary(decimals=3))
            print(result.summary)
            model.check_assumptions(train_df, p_value_threshold=0.05, show_plots=False)

        # plot coefficients with CI
        plot_coef_ci(model)

        # smoothed calibration curves, evaulating at time point 100
        survival_probability_calibration(model, train_df, t0=100)

        # predict survival probabilities
        survival = model.predict_survival_function(X_test, times=time_grid)
        surv_probs = survival.T.to_numpy()

        # plot survival curve per individuals
        # or sample_size=len(survival)
        plot_survival_functions(survival, sample_size=5)

        # # k fold cross validation using c-index
        # scores = k_fold_cross_validation(
        #     model,
        #     train_df,
        #     duration_col,
        #     event_col,
        #     k=5,
        #     scoring_method="concordance_index",
        #     seed=42,
        # )
        # print("K-Fold Score:", scores)
        # print("K-Fold Mean Score:", np.mean(scores))

        # c-index for training data
        print("Concordance Index on Training Set:", model.concordance_index_)

        # c-index on test data
        if model_name == "cox_lf":
            hazard_scores = model.predict_partial_hazard(X_test)

            # the usage:
            # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
            c_index = concordance_index_censored(
                y_test_surv[event_col], y_test_surv[duration_col], hazard_scores
            )
            print(f"Concordance Index on Test Set: {c_index[0]}")

            # get c-index based on inverse probability of censoring weights
            c_index_ipcw = concordance_index_ipcw(
                y_train_surv, y_test_surv, hazard_scores
            )
            print("Concordance Index IPCW on Test Set:", c_index_ipcw[0])

            # # c-index on test data (not necessary to have)
            # # get concordance index on test set
            # # if the predicted scores are risks/hazards, multiply by -1
            # # https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
            # c_index = concordance_index(
            #     test_df[duration_col], -hazard_scores, test_df[event_col]
            # )
            # print("Concordance Index on Test Set:", c_index)

        # c-index on test data
        print(
            "Concordance Index on Test Set:",
            model.score(test_df, scoring_method="concordance_index"),
        )

        # get ibs
        # estimate should be the survival probabilites
        ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_probs, time_grid)
        print("Integrated Brier Score:", ibs)

        # get time dependent auc
        risk_probs = 1 - surv_probs
        auc_scores, mean_auc_score = cumulative_dynamic_auc(
            y_train_surv, y_test_surv, risk_probs, event_time_grid
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

        # plot survival curve
        plot_survival_cols(model, train_df, cols=["age", "male"], bins_count=5)

        # aft models can show expected times
        if model_name in ["weibull", "ln", "ll"]:
            expected_survival = model.predict_expectation(X_test)
            plot_expected_survival(expected_survival, model_name)

        # to run emm, get new test df after fitting a model
        model_with_prob = test_df.copy()
        model_with_prob = get_avg_hourly(model_with_prob, survival, duration_col)
        save_parquet(model_with_prob, "./data_sa", f"{model_name}_with_prob.parquet")
