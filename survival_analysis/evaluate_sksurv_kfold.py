import numpy as np

from sklearn.model_selection import StratifiedKFold

from survival_analysis.fit import (
    fit_cox_sk,
    fit_coxnet_sk,
    fit_gb,
    fit_cgb,
    fit_rsf,
    fit_ersf,
)

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


def evaluate_sksurv_kfold(
    model_names,
    df,
    X,
    y,
    duration_col,
    event_col,
    n_splits=5,
    random_state=42,
    n_timepoints=50,
):

    # initializing stratified kfold and results dic
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for model_name in model_names:
        fit_function = FIT_FUNCTIONS2[model_name]

        # for individual score
        c_index = []
        c_index_censored = []
        c_index_ipcw = []
        ibs = []
        auc_mean = []

        # run the kfold and get results per model
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, df[event_col])):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]

            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

            # fit model on training
            model = fit_function(X_train_fold, y_train_fold)

            # predict risk score
            hazard_scores = model.predict(X_test_fold)

            # get time grids per fold in the test fold
            min_time_fold = df.iloc[test_idx][duration_col].min()
            max_time_fold = df.iloc[test_idx][duration_col].max()

            time_grid_fold = np.linspace(
                min_time_fold,
                max_time_fold - 1,
                num=n_timepoints,
                endpoint=True,
            )

            # event time grids
            event_times_fold = df.iloc[test_idx].loc[
                df.iloc[test_idx][event_col] == 1, duration_col
            ]
            if len(event_times_fold) > 0:
                min_event_time_fold = event_times_fold.min()
                max_event_time_fold = event_times_fold.max()
            else:
                min_event_time_fold = min_time_fold
                max_event_time_fold = max_time_fold

            event_time_grid_fold = np.linspace(
                min_event_time_fold,
                max_event_time_fold - 1,
                num=n_timepoints,
                endpoint=True,
            )

            # predict survival based on the time grid per fold
            step_funcs = model.predict_survival_function(
                X_test_fold, return_array=False
            )
            surv_probs = np.array([sf(time_grid_fold) for sf in step_funcs])

            # get c index
            c_index_fold = model.score(X_test_fold, y_test_fold)
            c_index.append(c_index_fold)

            # get c index (same)
            c_index_cens = concordance_index_censored(
                y_test_fold[event_col], y_test_fold[duration_col], hazard_scores
            )
            c_index_censored.append(c_index_cens[0])

            # get ipcw c-index
            c_index_ipcw_val = concordance_index_ipcw(
                y_train_fold, y_test_fold, hazard_scores
            )[0]
            c_index_ipcw.append(c_index_ipcw_val)

            # get integrated brier score
            ibs_val = integrated_brier_score(
                y_train_fold, y_test_fold, surv_probs, time_grid_fold
            )
            ibs.append(ibs_val)

            # time-dependent AUC using event time grid fold
            auc_scores, auc_mean_val = cumulative_dynamic_auc(
                y_train_fold, y_test_fold, surv_probs, event_time_grid_fold
            )
            auc_mean.append(auc_mean_val)

            # print(f"Kfold for model: {model_name}")
            # print(
            #     f"Fold {fold_idx+1}: "
            #     f"Concordance Index={c_index_fold}, "
            #     f"IBS={ibs_val}, "
            #     f"AUC={auc_mean}"
            # )

        results[model_name] = {
            "Mean Concordance Index": np.mean(c_index),
            # "Mean Concordance Index": np.mean(c_index_censored),
            "Mean IPCW C-index": np.mean(c_index_ipcw),
            "Mean Integrated Brier Score": np.mean(ibs),
            "Mean Time-Dependent AUC": np.mean(auc_mean),
        }

    return results
