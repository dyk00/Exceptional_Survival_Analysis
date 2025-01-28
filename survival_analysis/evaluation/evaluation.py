# only if it might be used multiple times

# lifelines
from lifelines.utils import concordance_index

# scikit-survival
from sksurv.metrics import concordance_index_ipcw

# get concordance index on test set
# if the predicted scores are risks/hazards, multiply by -1
# https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
def get_c_index_lifelines(test_df, duration_col, event_col, hazard_scores):
    return concordance_index(
        test_df[duration_col],
        -hazard_scores,
        test_df[event_col],
    )


# get the c index using sksurv package
def get_c_index_sksurv(estimator, X_test, y_test):
    return estimator.score(X_test, y_test)


# get concordance index based on inverse probability of censoring weights
def get_c_index_ipcw(y_train_surv, y_test_surv, hazard_scores):
    result = concordance_index_ipcw(y_train_surv, y_test_surv, hazard_scores)
    return result[0]
