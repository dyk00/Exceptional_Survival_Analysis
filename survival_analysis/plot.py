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


DIR = ".."

# plot coefficients with CI
def plot_coef_ci(model, model_name, output_dir=DIR):
    plt.figure(figsize=(6, 4))
    model.plot()
    plt.tight_layout()

    filename = f"{output_dir}/{model_name}_coef_ci.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


# plot first few samples (individuals) as step function (discrete time)
def plot_survival_functions(survival, model_name, sample_size=5, output_dir=DIR):
    plt.figure(figsize=(5, 5))
    for column in survival.columns[:sample_size]:
        plt.step(
            survival.index.to_numpy(),
            survival[column].to_numpy(),
            label=f"Individual {column}",
        )
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)

    filename = f"{output_dir}/{model_name}_survival_functions.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


# plot time dependent auc based on times with events
# to evaluate discriminative power
def plot_time_dependent_auc(
    event_time_grid, auc_scores, mean_auc_score, model_name, output_dir=DIR
):
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

    filename = f"{output_dir}/{model_name}_time_dependent_auc.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


# plot time dependent roc curve
# roc curve does not break on a normal time grid
# use grid because survival is computed based on defined time grid
def plot_time_dependent_roc(
    survival, time_grid, test_df, duration_col, event_col, model_name, output_dir=DIR
):
    test_df = test_df.reset_index(drop=True)
    if not survival.columns.is_unique:
        survival.columns = np.arange(survival.shape[1])

    plt.figure(figsize=(6, 4))
    sorted_times = np.sort(time_grid)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_times)))

    # for each time point,
    for i, t in enumerate(sorted_times):

        # get the risk score on the corresponding index
        event_probs = (1.0 - survival.loc[t]).reindex(test_df.index)

        # someone who might be at risk (censored) and already had by t
        duration_series = test_df[duration_col]
        event_series = test_df[event_col]

        label = (duration_series >= t) | (
            (duration_series <= t) & (event_series == True)
        )

        # had duration before and incl t, and had event
        valid_case = ((duration_series <= t) & event_series)[label]
        prob = event_probs[label]

        print(
            f"Time={t}, "
            f"#valid={label.sum()}, "
            f"#positives={valid_case.sum()}, "
            f"min(riskscore)={prob.min()}, "
            f"max(riskscore)={prob.max()}"
        )

        # compute the roc and auc
        fpr, tpr, _ = roc_curve(valid_case, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"t={t:.1f} (AUC={roc_auc:.7f})", color=colors[i])

    # reference line
    plt.plot([0, 1], [0, 1], "k--", label="Random   (AUC=0.500)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

    # plt.title("Time-Dependent ROC Curves upto Multiple Time Points")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

    filename = f"{output_dir}/{model_name}_time_dependent_roc.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_km(train_df, duration_col, event_col, strata=None, output_dir=DIR):
    km = KaplanMeierFitter()

    plt.figure(figsize=(6, 4))

    # plot different curves per strata
    if strata:
        groups = train_df[strata].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))
        for i, group in enumerate(groups):
            mask = train_df[strata] == group
            km.fit(
                train_df.loc[mask, duration_col],
                event_observed=train_df.loc[mask, event_col],
                label=str(group),
            )
            km.plot_survival_function(ci_show=True, color=colors[i])
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
    plt.xlim(0, 24)
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.legend(title=str(strata), bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

    filename = f"{output_dir}/km_plot_{strata}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


# specific to lifelines models
def plot_survival_cols(model, train_df, cols, bins_count, model_name, output_dir=DIR):
    plt.figure(figsize=(10, 4))

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
    ax = model.plot_partial_effects_on_outcome(cols, values, cmap="rainbow")

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
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Variables",
        ncol=3,
        fontsize="small",
    )

    # plt.title(f"Partial Effects of {', '.join(cols)} on Survival Outcome")
    ax.set_xlim(0, 24)
    plt.grid(True)
    # plt.tight_layout()
    filename = f"{output_dir}/{model_name}_{'_'.join(cols)}_survival_cols.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


# plot expected survival times for aft models
def plot_expected_survival(expected_survival, model_name, output_dir=DIR):
    plt.figure(figsize=(5, 4))
    data = pd.DataFrame({"Expected Survival Time": expected_survival.values})
    sns.boxplot(y="Expected Survival Time", data=data, color="skyblue")
    plt.title(f"Boxplot of Expected Survival Times ({model_name})")
    plt.ylabel("Expected Survival Time")
    plt.xticks([], [])
    plt.tight_layout()
    filename = f"{output_dir}/{model_name}_expected_survival.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# plot distribution of events over time
def plot_distribution_is_first_under24(
    under_t, duration_col, event_col, output_dir=DIR
):
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=under_t,
        x=duration_col,
        hue=event_col,
        bins=50,
        multiple="stack",
        palette="colorblind",
    )
    plt.title("Distribution of Time to First Event <24 Hours")
    plt.xlabel("Time to First Event (hours)")
    plt.ylabel("Count")
    plt.savefig(
        f"{output_dir}/distribution_is_first_under24.png", dpi=300, bbox_inches="tight"
    )
    plt.close("all")


def plot_distribution_is_first_entire(df, duration_col, event_col, output_dir=DIR):
    plt.figure(figsize=(8, 6))
    sns.histplot(
        data=df,
        x=duration_col,
        hue=event_col,
        bins=50,
        multiple="stack",
        palette="colorblind",
    )
    plt.title("Distribution of Time to First Event over Entire Hours")
    plt.xlabel("Time to First Event (hours)")
    plt.ylabel("Count")
    plt.savefig(
        f"{output_dir}/distribution_is_first_entire.png", dpi=300, bbox_inches="tight"
    )
    plt.close("all")


def plot_distribution_first_event_under24(
    under_t, duration_col, first_event_col="first_event", output_dir=DIR
):
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=under_t,
        x=duration_col,
        hue=first_event_col,
        bins=500,
        multiple="stack",
        palette="colorblind",
    )
    plt.title("Distribution of Time to First Event Type <24 Hours")
    plt.xlabel("Time to First Event Type (hours)")
    plt.ylabel("Count")
    plt.savefig(
        f"{output_dir}/distribution_first_event_under24.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


# plot the duration histogram
def plot_value_counts_histogram(df, column, output_dir=DIR):
    vc = df[column].value_counts().sort_index()
    print(vc.to_string())
    plt.figure(figsize=(8, 6))
    plt.bar(vc.index, vc.values, color="skyblue")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title("Histogram of Duration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/value_counts_hist.png", dpi=300, bbox_inches="tight")
    plt.close("all")
