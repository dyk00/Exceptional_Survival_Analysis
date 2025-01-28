from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

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


# predict hazard (risk) scores
def predict_hazard_sksurv(estimator, X_test):
    prediction = estimator.predict(X_test)
    return prediction
