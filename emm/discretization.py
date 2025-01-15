import pandas as pd

# apply equal frequency binning
# so that approximately equal number of values are in each bin

# note: you could use qcut, however when there are duplicates
# it does not assign approximately equal numbers in each bin even if you drop duplicates
def discretize_numeric_cols(df, columns, bins_count=2):

    # initialize total discretized variables
    col_bins = {}

    # discretize numeric columns
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):

            # just in case there are missing values
            values = df[col].dropna().tolist()
            values.sort()

            # length of values of each column
            n = len(values)

            # here we use equal frequency binning
            # the number of values that each bin can hold, where bins_count is given
            bin_size = n / bins_count

            # initalize bins for a column
            bins = []
            start_index = 0

            # loop through for each bin
            for b in range(bins_count):

                # if the current bin is the end bin, then set the end index as length
                if b == bins_count - 1:
                    end_index = n

                # otherwise, set each bin's end index
                # as the numbers of existing bins times the bin size
                else:
                    end_index = int(round((b + 1) * bin_size))

                # assign the sorted values to each bin based on indices
                bin_vals = values[start_index:end_index]

                # append the bin range to each bins, rather than actual values
                # e.g. {'age': (0, 10], (10, 20], (20, 30]}
                if len(bin_vals) > 0:
                    bins.append((bin_vals[0], bin_vals[-1]))

                # update start index and go on
                start_index = end_index

            # assign the bins to each column
            col_bins[col] = bins

    # return the whole discretized variables
    return col_bins
