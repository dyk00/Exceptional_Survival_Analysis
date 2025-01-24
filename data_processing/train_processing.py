import pandas as pd
from sklearn.model_selection import train_test_split


def index_data(df):
    df = df.set_index(["p_id", "admission_id"])
    df = df.reset_index(drop=True)
    return df


def split_data(df, duration_col, event_col, test_size=0.2, random_state=42):
    df_copy = df.copy()

    train_df, test_df = train_test_split(
        df_copy, test_size=test_size, random_state=random_state
    )

    X = df_copy.drop([event_col, duration_col], axis=1)
    y = df_copy[[event_col, duration_col]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return train_df, test_df, X_train, X_test, y_train, y_test
