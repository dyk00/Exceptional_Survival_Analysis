from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees

# fit random survival forest
def fit_rsf(
    X_train,
    y_train,
    n_estimators=100,
    random_state=42,
):
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rsf.fit(X_train, y_train)
    return rsf


# fit extremely random survival forest
def fit_ersf(
    X_train,
    y_train,
    n_estimators=100,
    random_state=42,
):
    ersf = RandomSurvivalForest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    ersf.fit(X_train, y_train)
    return ersf


# predict hazard (risk) scores
def predict_hazard_sksurv(estimator, X_test):
    prediction = estimator.predict(X_test)
    return prediction
