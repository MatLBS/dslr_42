import pandas as pd
from toolkit_dslr.math_utils import (calculate_count, calculate_mean,
                                     calculate_std, find_min, find_max,
                                     calculate_percentile, calculate_skewness,
                                     calculate_kurtosis)


def describe_column(column: pd.Series) -> pd.Series:
    """
    Describe a pandas Series by calculating various statistics.
    """
    column_described = pd.Series()
    column = column.dropna()

    count = calculate_count(column)
    mean = calculate_mean(column)
    var, std = calculate_std(column, mean)
    min = find_min(column)
    percentile_25 = calculate_percentile(column, 25)
    percentile_50 = calculate_percentile(column, 50)
    percentile_75 = calculate_percentile(column, 75)
    max = find_max(column)
    skew = calculate_skewness(column)
    kurt = calculate_kurtosis(column)
    column_described["count"] = count
    column_described["mean"] = mean
    column_described["var"] = var
    column_described["std"] = std
    column_described["min"] = min
    column_described["25%"] = percentile_25
    column_described["50%"] = percentile_50
    column_described["75%"] = percentile_75
    column_described["max"] = max
    column_described["skew"] = skew
    column_described["kurtosis"] = kurt

    return column_described


def describe(file: str) -> pd.DataFrame:
    """
    Prints the summary statistics of a DataFrame.
    """

    df = pd.read_csv(file)

    df = df.drop(['Index', 'Hogwarts House',
                  'First Name', 'Last Name',
                  'Birthday', "Best Hand"], axis=1)

    df_described = pd.DataFrame()
    for i in df.columns:
        df_described[i] = describe_column(df[i])
    return df_described
