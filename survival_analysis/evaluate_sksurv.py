import numpy as np
import pandas as pd

from data_processing.data_io import save_parquet
from data_processing.surv_util import get_avg_hourly

from survival_analysis.fit import (
    fit_cox_sk,
    fit_coxnet_sk,
    fit_gb,
    fit_cgb,
    fit_rsf,
    fit_ersf,
)
from survival_analysis.plot import (
    # plot_coef_ci,  # not applicable
    plot_survival_functions,
    plot_time_dependent_auc,
    plot_time_dependent_roc,
    # plot_survival_cols, # not applicable
)

# from lifelines.utils import k_fold_cross_validation # not applicable
# from lifelines.calibration import survival_probability_calibration # not applicable

from sksurv.metrics import (
    integrated_brier_score,
    cumulative_dynamic_auc,
    concordance_index_censored,
    concordance_index_ipcw,
)

FIT_FUNCTIONS2 = {
    "cox_sk": fit_cox_sk,
    "coxnet_sk": fit_coxnet_sk,
    "gb": fit_gb,
    "gcb": fit_cgb,
    "rsf": fit_rsf,
    "ersf": fit_ersf,
}


def evaluate_sksurv(
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
    event_col,
    duration_col,
    time_grid_train,
    event_time_grid_train,
    time_grid_test,
    event_time_grid_test,
    time_grid_val,
    event_time_grid_val,
):
    global FIT_FUNCTIONS2

    for model_name in model_names:
        print(f"Evaluating : {model_name}")

        # lookup model name
        fit_function = FIT_FUNCTIONS2[model_name]

        # fit the model
        model = fit_function(X_train, y_train_surv)

        # predict hazard (risk) scores
        hazard_train = model.predict(X_train)
        hazard_val = model.predict(X_val)
        hazard_test = model.predict(X_test)

        #         # get a list of StepFunctions (n_samples,)
        #         step_funcs = model.predict_survival_function(X_test, return_array=False)

        #         # map each StepFunction on time_grid
        #         # shape (n_samples, n_timepoints)
        #         surv_probs = np.array([sf(time_grid) for sf in step_funcs])

        #         # shape = (n_timepoints,  n_samples)
        #         survival = pd.DataFrame(
        #             data=surv_probs.T,
        #             index=time_grid,
        #             columns=X_test.index,
        #         )

        step_funcs_train = model.predict_survival_function(X_train, return_array=False)
        surv_probs_train = np.array([sf(time_grid_train) for sf in step_funcs_train])
        survival_train = pd.DataFrame(
            data=surv_probs_train.T, index=time_grid_train, columns=X_train.index
        )

        step_funcs_val = model.predict_survival_function(X_val, return_array=False)
        surv_probs_val = np.array([sf(time_grid_val) for sf in step_funcs_val])
        survival_val = pd.DataFrame(
            data=surv_probs_val.T, index=time_grid_val, columns=X_val.index
        )

        step_funcs_test = model.predict_survival_function(X_test, return_array=False)
        surv_probs_test = np.array([sf(time_grid_test) for sf in step_funcs_test])
        survival_test = pd.DataFrame(
            data=surv_probs_test.T, index=time_grid_test, columns=X_test.index
        )

        # get c-index
        c_index_train = model.score(X_train, y_train_surv)
        print(f"Concordance Index on Training Set: {c_index_train}")
        c_index_val = model.score(X_val, y_val_surv)
        print(f"Concordance Index on Validation Set: {c_index_val}")
        c_index_test = model.score(X_test, y_test_surv)
        print(f"Concordance Index on Test Set: {c_index_test}")

        #         # the usage:
        #         # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
        #         c_index_sk = concordance_index_censored(
        #             y_test_surv[event_col], y_test_surv[duration_col], hazard_scores
        #         )
        #         print(f"Concordance Index on Test Set: {c_index_sk[0]}")

        # get c-index based on inverse probability of censoring weights
        c_index_ipcw = concordance_index_ipcw(y_train_surv, y_train_surv, hazard_train)
        print("Concordance Index IPCW on Training Set:", c_index_ipcw[0])

        c_index_ipcw = concordance_index_ipcw(y_train_surv, y_val_surv, hazard_val)
        print("Concordance Index IPCW on Validation Set:", c_index_ipcw[0])

        c_index_ipcw = concordance_index_ipcw(y_train_surv, y_test_surv, hazard_test)
        print("Concordance Index IPCW on Test Set:", c_index_ipcw[0])

        # get IBS
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

        # plot time-dependent AUC
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

        # plot partial effects not supported for scikit-survival

        #         # to run emm, get new test df after fitting a model
        #         model_with_prob = test_df.copy()
        #         model_with_prob = get_avg_hourly(model_with_prob, survival_test, duration_col)
        #         save_parquet(model_with_prob, "/home/u839129/escobar-working-area/Dayeong/data_sa_rand42", f"{model_name}_with_prob.parquet")

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
        step_funcs_test_grouped = model.predict_survival_function(
            X_test_grouped, return_array=False
        )
        surv_probs_test_grouped = np.array(
            [sf(time_grid_test) for sf in step_funcs_test_grouped]
        )
        survival_test_grouped = pd.DataFrame(
            data=surv_probs_test_grouped.T,
            index=time_grid_test,
            columns=X_test_grouped.index,
        )

        # get the avg and hourly surv probabilities
        model_with_prob_grouped = grouped_test_df.copy()
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
