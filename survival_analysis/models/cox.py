# lifelines
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# scikit-survival
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

# cox training using scikit-survival
# alpha being strength of regularization == penalizer in lifelines
def fit_cox_sksurv(X_train, y_train_surv, alpha_min_ratio=0.05, l1_ratio=0.5):
    coxph = CoxnetSurvivalAnalysis(
        alpha_min_ratio=alpha_min_ratio, l1_ratio=l1_ratio, fit_baseline_model=True
    )
    coxph.fit(X_train, y_train_surv)
    return coxph


def predict_hazard_cox_sksurv(coxph, X_test):
    prediction = coxph.predict(X_test)
    return prediction


def predict_probability_cox_sksurv(coxph_sksurv, X_test):
    return coxph_sksurv.predict_survival_function(X_test)


# cox training using lifelines
# alpha being CI of coef
def fit_cox_lifelines(
    train_df, duration_col, event_col, alpha=0.05, penalizer=0.01, l1_ratio=0.5
):
    cph = CoxPHFitter(alpha=alpha, penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(train_df, duration_col, event_col, batch_mode=True)
    return cph


# predict survival probability S(t) (prob of no event occured (event-free) by t)
# failure probability S(t) = 1- F(t) (prob of event occured by t)
def predict_probability_cox_lifelines(cph, X_test, times):
    surv_prob = cph.predict_survival_function(X_test, times=times)
    return surv_prob


# predict relative risk/hazard ratio for each individual
# so how much greater one’s hazard is relative to another
# but not probability
def predict_hazard_cox_lifelines(cph, X_test):
    return cph.predict_partial_hazard(X_test)


# test proportional hazards
def test_proportional_hazards(cph, df):
    return proportional_hazard_test(cph, df, time_transform="rank")


# check the assumptions and plot
def check_cox_assumptions(cph, train_df, thr=0.05, bool=False):
    cph.check_assumptions(train_df, p_value_threshold=thr, show_plots=bool)
