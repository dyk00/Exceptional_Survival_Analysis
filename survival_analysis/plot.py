# basic python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import product

# emm
from emm.discretization import discretize_numeric_cols

# lifelines
from lifelines import KaplanMeierFitter

# roc curve
from sklearn.metrics import roc_curve, auc

# plot coefficients with CI
def plot_coef_ci(model):
    plt.figure(figsize=(6, 4))
    model.plot()
    plt.tight_layout()
    plt.show()


# plot first few samples (individuals) as step function (discrete time)
def plot_survival_functions(survival, sample_size=5):
    plt.figure(figsize=(5, 5))
    for column in survival.columns[:sample_size]:
        plt.step(survival.index, survival[column], label=f"Individual {column}")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


# plot time dependent auc based on times with events
# to evaluate discriminative power
def plot_time_dependent_auc(event_time_grid, auc_scores, mean_auc_score):
    plt.figure(figsize=(6, 4))
    plt.plot(event_time_grid, auc_scores, marker="o", label="Time-dependent AUC")

    # add a line for mean auc
    plt.axhline(
        mean_auc_score,
        linestyle="--",
        color="red",
        label=f"Mean AUC = {mean_auc_score:.3f}",
    )
    plt.xlabel("Time Since Admission")
    plt.ylabel("Time-dependent AUC")

    # plt.title("AUC over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


# plot time dependent roc curve
# roc curve does not break on a normal time grid
# use grid because survival is computed based on defined time grid
def plot_time_dependent_roc(
    eval_times, survival, time_grid, test_df, duration_col, event_col
):
    plt.figure(figsize=(6, 4))

    # for each evaluation time point,
    for t in eval_times:

        # compare the closest time in the time grid and the last index
        # and take the minimum as index
        idx = min(np.searchsorted(time_grid, t), len(time_grid) - 1)
        closest_time = time_grid[idx]

        # as event means deterioration event in our case,
        # get the risk score on the corresponding index

        event_probs = 1 - survival.iloc[idx, :].values

        # (time_to_event <= closest_time) and (event == True)
        label = (
            (test_df[duration_col].values <= closest_time)
            & (test_df[event_col].values == True)
        ).astype(bool)

        # compute the roc and auc
        fpr, tpr, _ = roc_curve(label, event_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"t={closest_time:.2f} (AUC={roc_auc:.3f})")

    # reference line
    plt.plot([0, 1], [0, 1], "k--", label="Random   (AUC=0.500)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

    # plt.title("Time-Dependent ROC Curves upto Multiple Time Points")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()


def plot_km(train_df, duration_col, event_col, strata=None):
    km = KaplanMeierFitter()

    plt.figure(figsize=(6, 4))

    # plot different curves per strata
    if strata:
        groups = train_df[strata].unique()
        for group in groups:
            mask = train_df[strata] == group
            km.fit(
                train_df.loc[mask, duration_col],
                event_observed=train_df.loc[mask, event_col],
                label=str(group),
            )
            km.plot_survival_function(ci_show=True)
        # plt.title(f"Kaplan-Meier Survival Curves by {strata}")
    else:
        # single curve
        km.fit(
            train_df[duration_col],
            event_observed=train_df[event_col],
            label="Kaplan-Meier Estimate",
        )
        km.plot_survival_function(ci_show=True)
        # plt.title("Kaplan-Meier Survival Curve")

    plt.xlabel("Hours")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.legend()
    plt.show()


# specific to lifelines models
def plot_survival_cols(cph, train_df, cols, bins_count):
    plt.figure(figsize=(6, 4))

    # get the discretized column bins
    col_bins = discretize_numeric_cols(train_df, cols, bins_count=bins_count)

    # if given string, then convert it as list
    if isinstance(cols, str):
        cols = [cols]

    # dict for storing values for each bin
    value = {}
    for col in cols:

        # if bool, then get F/T
        if train_df[col].dtype == "bool":
            value[col] = [False, True]
        else:
            # if numeric, get the values of the bins
            bin_ranges = col_bins.get(col)

            # we will only use lower bound value
            value[col] = [b[0] for b in bin_ranges]

    # get the combinations of values
    combs = list(product(*value.values()))
    values = [list(comb) for comb in combs]

    # get the partial effect of the combinations
    ax = cph.plot_partial_effects_on_outcome(cols, values, cmap="tab20")

    # get lines
    lines = ax.get_lines()

    # for each result and values,
    for line, comb in zip(lines, combs):
        labels = []

        # for each column and values,
        for col, val in zip(cols, comb):

            # if bool, then append T/F to labels
            if isinstance(val, bool):
                labels.append(f"{col}={'True' if val else 'False'}")
            else:
                # if numeric, append rounded value
                labels.append(f"{col}={val:.1f}")

        # merge labels
        label = ", ".join(labels)
        line.set_label(label)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Variables",
    )

    # plt.title(f"Partial Effects of {', '.join(cols)} on Survival Outcome")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# plot expected survival times for aft models
def plot_expected_survival(expected_survival, model):
    plt.figure(figsize=(6, 6))
    data = pd.DataFrame({"Expected Survival Time": expected_survival.values})
    sns.boxplot(y="Expected Survival Time", data=data, color="skyblue")
    plt.title(f"Boxplot of Expected Survival Times ({model})")
    plt.ylabel("Expected Survival Time")
    plt.xticks([], [])
    plt.tight_layout()
    plt.show()
