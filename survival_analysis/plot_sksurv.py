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


def plot_sksurv(
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

        step_funcs_test = model.predict_survival_function(X_test, return_array=False)
        surv_probs_test = np.array([sf(time_grid_test) for sf in step_funcs_test])
        survival_test = pd.DataFrame(
            data=surv_probs_test.T, index=time_grid_test, columns=X_test.index
        )

        plot_survival_functions(survival_test, model_name=model_name, sample_size=5)
