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
            train_df = train_df.reset_index()
            val_df = val_df.reset_index()
            test_df = test_df.reset_index()
            train_df_fit = train_df.drop(columns=["p_id", "opname_id"])

            model = fit_function(train_df_fit, duration_col, event_col)

        # print the summary
        print(model.print_summary())

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

        # c index on all sets
        c_index_train = model.score(train_df, scoring_method="concordance_index")
        print(f"Concordance Index on Training Set: {c_index_train}")
        c_index_val = model.score(val_df, scoring_method="concordance_index")
        print(f"Concordance Index on Validation Set: {c_index_val}")
        c_index_test = model.score(test_df, scoring_method="concordance_index")
        print(f"Concordance Index on Test Set: {c_index_test}")

        # c-index on test data
        if model_name == "cox_lf":
            hazard_train = model.predict_partial_hazard(X_train)
            hazard_val = model.predict_partial_hazard(X_val)
            hazard_test = model.predict_partial_hazard(X_test)

            # get c-index based on inverse probability of censoring weights
            c_index_ipcw = concordance_index_ipcw(
                y_train_surv, y_train_surv, hazard_train
            )
            print("Concordance Index IPCW on Training Set:", c_index_ipcw[0])

            c_index_ipcw = concordance_index_ipcw(y_train_surv, y_val_surv, hazard_val)
            print("Concordance Index IPCW on Validation Set:", c_index_ipcw[0])

            c_index_ipcw = concordance_index_ipcw(
                y_train_surv, y_test_surv, hazard_test
            )
            print("Concordance Index IPCW on Test Set:", c_index_ipcw[0])

        ibs_train = integrated_brier_score(
            y_train_surv, y_train_surv, surv_probs_train, time_grid_train
        )
        ibs_val = integrated_brier_score(
            y_train_surv, y_val_surv, surv_probs_val, time_grid_val
        )
        ibs_test = integrated_brier_score(
            y_train_surv, y_test_surv, surv_probs_test, time_grid_test
        )

        print(f"Integrated Brier Score on Training Set: {ibs_train}")
        print(f"Integrated Brier Score on Validation Set: {ibs_val}")
        print(f"Integrated Brier Score on Test Set: {ibs_test}")

        risk_probs_train = 1 - surv_probs_train
        risk_probs_val = 1 - surv_probs_val
        risk_probs_test = 1 - surv_probs_test

        auc_scores_train, mean_auc_train = cumulative_dynamic_auc(
            y_train_surv, y_train_surv, risk_probs_train, event_time_grid_train
        )
        auc_scores_val, mean_auc_val = cumulative_dynamic_auc(
            y_train_surv, y_val_surv, risk_probs_val, event_time_grid_val
        )
        auc_scores_test, mean_auc_test = cumulative_dynamic_auc(
            y_train_surv, y_test_surv, risk_probs_test, event_time_grid_test
        )

        print(f"Mean AUC on Training Set: {mean_auc_train}")
        print(f"Mean AUC on Validation Set: {mean_auc_val}")
        print(f"Mean AUC on Test Set: {mean_auc_test}")

        plot_time_dependent_auc(
            event_time_grid_train,
            auc_scores_train,
            mean_auc_train,
            model_name=model_name + "_train",
        )
        plot_time_dependent_auc(
            event_time_grid_val,
            auc_scores_val,
            mean_auc_val,
            model_name=model_name + "_val",
        )
        plot_time_dependent_auc(
            event_time_grid_test,
            auc_scores_test,
            mean_auc_test,
            model_name=model_name + "_test",
        )

        plot_time_dependent_roc(
            survival=survival_train,
            time_grid=time_grid_train,
            test_df=train_df,
            duration_col=duration_col,
            event_col=event_col,
            model_name=model_name + "_train",
        )

        plot_time_dependent_roc(
            survival=survival_val,
            time_grid=time_grid_val,
            test_df=val_df,
            duration_col=duration_col,
            event_col=event_col,
            model_name=model_name + "_val",
        )

        plot_time_dependent_roc(
            survival=survival_test,
            time_grid=time_grid_test,
            test_df=test_df,
            duration_col=duration_col,
            event_col=event_col,
            model_name=model_name + "_test",
        )

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

        #         # aft models can show expected times
        #         if model_name in ["weibull", "ln", "ll"]:
        #             expected_survival = model.predict_expectation(X_test)
        #             plot_expected_survival(expected_survival, model_name=model_name)

        # code for below, for calculating the avg surv and hourly probs to each opname
        test_df_reset = test_df.reset_index(drop=False)

        # group per p_id and opname_id
        grouped_test_df = test_df_reset.groupby(
            ["p_id", "opname_id"], as_index=False
        ).first()

        # drop unnecessary cols for predicting
        X_test_grouped = grouped_test_df.drop(
            columns=["p_id", "opname_id", "time_to_first_event", "is_first"],
            errors="ignore",
        )

        # get the prediction
        survival_test_grouped = model.predict_survival_function(
            X_test_grouped, times=time_grid_test
        )
        model_with_prob_grouped = grouped_test_df.copy()

        # get the avg and hourly surv probabilities
        model_with_prob_grouped = get_avg_hourly(
            model_with_prob_grouped,
            survival_test_grouped,
            duration_col="time_to_first_event",
        )

        test_df_reset = test_df.reset_index(drop=False)

        # merge back with p id and opname id
        final_test_df = test_df_reset.merge(
            model_with_prob_grouped[
                [
                    "p_id",
                    "opname_id",
                    "avg_survival_probability",
                    "hourly_probabilities",
                ]
            ],
            on=["p_id", "opname_id"],
            how="left",
        )
        save_parquet(
            final_test_df,
            "..",
            f"{model_name}_with_prob.parquet",
        )


#         # checking if the same opname id indeed have the same probs
#         unique_counts = final_test_df.groupby('opname_id')['avg_survival_probability'].nunique()
#         print(unique_counts)
#         print((unique_counts == 1).all())

#         unique_counts1 = final_test_df.groupby('opname_id')['hourly_probabilities_tuple'].nunique()
#         print(unique_counts1)
#         print((unique_counts1 == 1).all())
