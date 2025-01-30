import numpy as np
import pandas as pd

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
    test_df,
    X_train,
    X_test,
    y_train_surv,
    y_test_surv,
    event_col,
    duration_col,
    time_grid,
    event_time_grid,
):
    global FIT_FUNCTIONS2

    for model_name in model_names:
        print(f"Evaluating : {model_name}")

        # lookup model name
        fit_function = FIT_FUNCTIONS2[model_name]

        # fit the model
        model = fit_function(X_train, y_train_surv)

        # predict hazard (risk) scores
        hazard_scores = model.predict(X_test)

        # get a list of StepFunctions (n_samples,)
        step_funcs = model.predict_survival_function(X_test, return_array=False)

        # map each StepFunction on time_grid
        # shape (n_samples, n_timepoints)
        surv_probs = np.array([sf(time_grid) for sf in step_funcs])

        # shape = (n_timepoints,  n_samples)
        survival = pd.DataFrame(data=surv_probs.T, index=time_grid)
        plot_survival_functions(survival, sample_size=5)

        # get c-index
        gb_c_index = model.score(X_test, y_test_surv)
        print("Concordance Index:", gb_c_index)

        # the usage:
        # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
        c_index_sk = concordance_index_censored(
            y_test_surv[event_col], y_test_surv[duration_col], hazard_scores
        )
        print(f"Concordance Index on Test Set: {c_index_sk[0]}")

        # get c-index based on inverse probability of censoring weights
        c_index_ipcw = concordance_index_ipcw(y_train_surv, y_test_surv, hazard_scores)[
            0
        ]
        print("IPCW Concordance Index:", c_index_ipcw)

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

        # plot partial effects not supported for scikit-survival
