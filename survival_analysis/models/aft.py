from lifelines import (
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
)


def fit_weibull(train_df, duration_col, event_col):
    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(train_df, duration_col=duration_col, event_col=event_col)
    return aft


def fit_ln(train_df, duration_col, event_col):
    lnf = LogNormalAFTFitter(penalizer=0.01)
    lnf.fit(train_df, duration_col=duration_col, event_col=event_col)
    return lnf


def fit_ll(train_df, duration_col, event_col):
    aft = LogLogisticAFTFitter(penalizer=0.01)
    aft.fit(train_df, duration_col=duration_col, event_col=event_col)
    return aft
