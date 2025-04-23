import seaborn as sns
import matplotlib.pyplot as plt

from data_processing.data_io import save_parquet

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def fit_lr(X_train, y_train, duration_col, event_col):
    # make y_train binary
    y_train = y_train.drop(columns=[duration_col])
    y_train = y_train[event_col].astype(int)

    # fit with class weights, replicating less frequent events
    lr = LogisticRegression(
        random_state=42,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        max_iter=50000,
    )
    lr.fit(X_train, y_train)

    return lr


# predict probabilities of event=0 or event=1 per row
# [p(event=0), p(event=1)]
# return only survival probability
def predict_lr(lr, X_test, test_df):
    # predict labels
    y_pred = lr.predict(X_test)

    # predict probabilities for event=False (surviving case)
    y_pred_prob = lr.predict_proba(X_test)[:, 0]
    # y_pred_prob = lr.predict_proba(X_test)[:, 1]

    # add a single probability
    model_with_prob = test_df.copy()
    model_with_prob["lr_probability"] = y_pred_prob
    save_parquet(
        model_with_prob,
        "..",
        "lr_with_prob.parquet",
    )
    lr_test_df = model_with_prob

    return y_pred, y_pred_prob, lr_test_df


def evaluate_lr(lr, X_test, y_test, test_df, event_col):
    # predict labels and probabilities
    y_pred, y_pred_prob, _ = predict_lr(lr, X_test, test_df)

    # typecast y_true and y_pred as binary
    y_true = y_test[event_col].astype(int)
    y_pred = y_pred.astype(int)

    # evaluation metrics for event = False (surviving case)
    # TP+FP and/or TP+FN are 0 (i.e., undefined), then set value to 0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    class_report = classification_report(y_true, y_pred, zero_division=0)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    print("Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    class_report = classification_report(y_true, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(class_report)

    return y_true, y_pred, y_pred_prob


# plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


# plot roc curve
def plot_roc_curve(y_true, y_pred_prob):
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="blue")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Roc Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
