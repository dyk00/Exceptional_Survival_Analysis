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


def plot_lifelines(
    model_names,
    train_df,
    test_df,
    val_df,
    X_train,
    X_test,
    X_val,
    y_train_surv,
    y_test_surv,
    y_val_surv,
    duration_col,
    event_col,
    time_grid_train,
    event_time_grid_train,
    time_grid_test,
    event_time_grid_test,
    time_grid_val,
    event_time_grid_val,
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

        # print the proportional hazards summary and check assumptions
        #         if model_name == "cox_lf":
        #             print("Proportional Hazards Summary:\n")
        #             result = proportional_hazard_test(model, train_df, time_transform="rank")
        #             print(result.print_summary(decimals=3))
        #             print(result.summary)
        #             model.check_assumptions(train_df, p_value_threshold=0.05, show_plots=False)

        survival_train = model.predict_survival_function(X_train, times=time_grid_train)
        surv_probs_train = survival_train.T.to_numpy()

        survival_val = model.predict_survival_function(X_val, times=time_grid_val)
        surv_probs_val = survival_val.T.to_numpy()

        survival_test = model.predict_survival_function(X_test, times=time_grid_test)
        surv_probs_test = survival_test.T.to_numpy()

        # plot survival curve
        plot_survival_cols(
            model, train_df, cols=["geslacht"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model, train_df, cols=["leeftijd"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model, train_df, cols=["first_event"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model, train_df, cols=["spoed"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["count_acute_ic"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["count_death_fullcode"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["count_death_ic"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model, train_df, cols=["count_ic_6hr"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["hoofdverrichting_code"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["specialisme_code"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model, train_df, cols=["m_day"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model, train_df, cols=["m_hour"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model, train_df, cols=["m_month"], bins_count=10, model_name=model_name
        )
        plot_survival_cols(
            model, train_df, cols=["m_year"], bins_count=10, model_name=model_name
        )

        plot_survival_cols(
            model,
            train_df,
            cols=["first_event", "geslacht"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["leeftijd", "geslacht"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["hoofdverrichting_code", "geslacht"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["specialisme_code", "geslacht"],
            bins_count=10,
            model_name=model_name,
        )

        plot_survival_cols(
            model,
            train_df,
            cols=["first_event", "leeftijd"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["first_event", "specialisme_code"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["first_event", "hoofdverrichting_code"],
            bins_count=10,
            model_name=model_name,
        )

        plot_survival_cols(
            model,
            train_df,
            cols=["hoofdverrichting_code", "leeftijd"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["specialisme_code", "leeftijd"],
            bins_count=10,
            model_name=model_name,
        )

        plot_survival_cols(
            model,
            train_df,
            cols=["first_event", "geslacht", "leeftijd"],
            bins_count=10,
            model_name=model_name,
        )

        plot_survival_cols(
            model,
            train_df,
            cols=["first_event", "hoofdverrichting_code", "geslacht", "leeftijd"],
            bins_count=10,
            model_name=model_name,
        )
        plot_survival_cols(
            model,
            train_df,
            cols=["first_event", "specialisme_code", "geslacht", "leeftijd"],
            bins_count=10,
            model_name=model_name,
        )

        # aft models can show expected times
        if model_name in ["weibull", "ln", "ll"]:
            expected_survival = model.predict_expectation(X_test)
            plot_expected_survival(expected_survival, model_name=model_name)

        # plot coefficients with CI
        plot_coef_ci(model, model_name=model_name)

        # plot survival curve per individuals
        # or sample_size=len(survival)
        plot_survival_functions(survival_test, model_name=model_name, sample_size=5)
