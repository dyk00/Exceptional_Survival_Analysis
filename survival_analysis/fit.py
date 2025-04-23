# lifelines
from lifelines import (
    CoxPHFitter,
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
)

# scikit-survival
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    ComponentwiseGradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
    ExtraSurvivalTrees,
)

# cox training using lifelines
# alpha being CI of coef
def fit_cox_lf(train_df, duration_col, event_col, alpha=0.05, penalizer=0.01):
    cph = CoxPHFitter(alpha=alpha, penalizer=penalizer)
    cph.fit(train_df, duration_col, event_col, batch_mode=True)
    return cph


# cox training using scikit-survival
def fit_cox_sk(X_train, y_train_surv, alpha=0.01):
    coxph = CoxPHSurvivalAnalysis(alpha=alpha)
    coxph.fit(X_train, y_train_surv)
    return coxph


# cox training using scikit-survival
# alpha being strength of regularization == penalizer in lifelines
def fit_coxnet_sk(X_train, y_train_surv, alphas=[0.01], l1_ratio=0.5):
    coxphnet = CoxnetSurvivalAnalysis(
        alphas=alphas, l1_ratio=l1_ratio, fit_baseline_model=True
    )
    coxphnet.fit(X_train, y_train_surv)
    return coxphnet


def fit_weibull(train_df, duration_col, event_col):
    weibull = WeibullAFTFitter(penalizer=0.01)
    weibull.fit(train_df, duration_col=duration_col, event_col=event_col)
    return weibull


def fit_ln(train_df, duration_col, event_col):
    ln = LogNormalAFTFitter(penalizer=0.01)
    ln.fit(train_df, duration_col=duration_col, event_col=event_col)
    return ln


def fit_ll(train_df, duration_col, event_col):
    ll = LogLogisticAFTFitter(penalizer=0.01)
    ll.fit(train_df, duration_col=duration_col, event_col=event_col)
    return ll


# fit gradient boosting
def fit_gb(X_train, y_train, learning_rate=1, n_estimators=100, random_state=42):
    gb = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    gb.fit(X_train, y_train)
    return gb


# fit componentwise gradient boosting
def fit_cgb(X_train, y_train, learning_rate=1, n_estimators=100, random_state=42):
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    cgb.fit(X_train, y_train)
    return cgb


# fit random survival forest
def fit_rsf(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42,
):
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=False,
        # oob_score=True
    )
    rsf.fit(X_train, y_train)
    return rsf


# fit extremely random survival forest
def fit_ersf(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    random_state=42,
):
    ersf = ExtraSurvivalTrees(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=False,
        # oob_score=True
    )
    ersf.fit(X_train, y_train)
    return ersf
