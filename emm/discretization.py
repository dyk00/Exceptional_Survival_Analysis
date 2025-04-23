# this is for global discretization. Will not be used for EMM.

import pandas as pd
import math

# feature that will be binned into single values
single_binned = [
    "geslacht",
    "spoed",
    "specialisme_code",
    "first_event",
    "count_death_fullcode",
    "count_death_ic",
    "count_ic_6hr",
    "count_acute_ic",
    "hoofdverrichting_code",
    "prioriteit_code",
    "care_order",
    "m_year",
    "m_month",
    "m_day",
    "m_hour",
]

# apply fixed value, equal width binning
def discretize_numeric_cols(df, columns, bins_count=2, bin_value=None):

    # initialize total discretized variables
    col_bins = {}

    # for some variables, we just make a bin for each value
    for col in columns:
        if col in single_binned or pd.api.types.is_bool_dtype(df[col]):
            values = sorted(df[col].unique().tolist())
            col_bins[col] = [(val,) for val in values]

        else:
            # discretize numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):

                # just in case there are missing values
                values = df[col].dropna()

                # get min and max values
                min_val = values.min()
                max_val = values.max()

                # initalize bins
                bins = []

                # if fixed bin width is given,
                if bin_value is not None:

                    # for handling min value,
                    # if the minimum value is multiples of bin width
                    if min_val % bin_value == 0:

                        # assign the current edge to minimum value
                        c_edge = min_val

                    # otherwise set next edge as (min value + next possible multiple value)
                    else:
                        n_edge = math.ceil(min_val / bin_value) * bin_value
                        bins.append((min_val, n_edge))

                        # update current edge to next edge
                        c_edge = n_edge

                    # for intervals,
                    # if the upper limit of a bin is within the range
                    while c_edge + bin_value <= max_val:

                        # keep appending the tuple by increasing bin_value to the current edge
                        bins.append((c_edge, c_edge + bin_value))

                        # update the next current edge
                        c_edge = c_edge + bin_value

                    # for handling max,
                    # if max value doesn't fit with bins
                    if c_edge < max_val:

                        # append with the current edge and max value
                        bins.append((c_edge, max_val))

                # if no bin_value is given,
                else:

                    # each bin width
                    bin_width = (max_val - min_val) / bins_count

                    # loop through for each bin
                    for b in range(bins_count):

                        # the lower bound of bin
                        l = min_val + b * bin_width

                        # for the last bin, the max value is the higher bound
                        if b == bins_count - 1:
                            r = max_val

                        # otherwise, set higher bound as lower + width
                        else:
                            r = l + bin_width

                        # append for each bin
                        bins.append((l, r))

                # assign the bins to each column
                col_bins[col] = bins

    #         print(f"For column", col)
    #         print(col_bins[col], "\n")

    # return the whole discretized variables
    return col_bins
