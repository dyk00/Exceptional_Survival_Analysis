import numpy as np

from survival_analysis.fit import fit_cox_lf, fit_weibull, fit_ln, fit_ll

from sklearn.model_selection import StratifiedKFold

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


def evaluate_lifelines_kfold(
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
        fit_function = FIT_FUNCTIONS1[model_name]

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

            train_fold_df = df.iloc[train_idx]
            test_fold_df = df.iloc[test_idx]

            if model_name == "cox_lf":
                model = fit_function(
                    train_fold_df,
                    duration_col,
                    event_col,
                    alpha=0.05,
                    penalizer=0.01,
                )
            else:
                model = fit_function(
                    train_fold_df,
                    duration_col,
                    event_col,
                )

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
            survival = model.predict_survival_function(
                X_test_fold, times=time_grid_fold
            )
            surv_probs = survival.T.to_numpy()

            # c-index on test data
            if model_name == "cox_lf":
                hazard_scores = model.predict_partial_hazard(X_test_fold)

                # the usage:
                # https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
                c_index_cens = concordance_index_censored(
                    y_test_fold[event_col], y_test_fold[duration_col], hazard_scores
                )
                c_index_censored.append(c_index_cens[0])

                # get c-index based on inverse probability of censoring weights
                c_index_ipcw_val = concordance_index_ipcw(
                    y_train_fold, y_test_fold, hazard_scores
                )
                c_index_ipcw.append(c_index_ipcw_val[0])

            # c-index on test data
            c_index_val = (
                model.score(test_fold_df, scoring_method="concordance_index"),
            )
            c_index.append(c_index_val)

            # get ibs
            # estimate should be the survival probabilites
            ibs_val = integrated_brier_score(
                y_train_fold, y_test_fold, surv_probs, event_time_grid_fold
            )
            ibs.append(ibs_val)

            # get time dependent auc
            auc_scores, mean_auc_score = cumulative_dynamic_auc(
                y_train_fold, y_test_fold, surv_probs, event_time_grid_fold
            )
            auc_mean.append(mean_auc_score)

            results[model_name] = {
                "Mean Concordance Index": np.mean(c_index),
                "Mean Concordance Index (censored)": np.mean(c_index_censored)
                if c_index_censored
                else np.nan,
                "Mean IPCW C-index": np.mean(c_index_ipcw) if c_index_ipcw else np.nan,
                "Mean Integrated Brier Score": np.mean(ibs),
                "Mean Time-Dependent AUC": np.mean(auc_mean),
            }

    return results
